"""Lambda handler for the WebSocket $default route.

Flow:
  1. Validate the envelope (auth.validate_envelope_bytes).
  2. Branch on message type:
       - server.register   → write RelaySessions row, store server_id ↔ connection.
       - server.heartbeat  → bump last_seen for the registered server.
       - client.attach     → bind connection to an existing server_id (client role).
       - worker.attach     → bind connection to an existing server_id (worker role).
       - other types       → route to the registered server connection.
  3. On failure, send an `error` envelope back to the originating
     connection. On success, forward to the routing target.

Important: the Lambda never opens the user PC's port. It only forwards
between WebSocket connections that the server/clients/workers already
opened *outbound* to API Gateway.
"""

from __future__ import annotations

import base64
import json
import os
import time

import boto3
from boto3.dynamodb.conditions import Key

from auth import (
    RelayLimits,
    authorize_attach,
    authorize_register,
    make_error,
    route,
    validate_envelope_bytes,
)

CONNECTIONS_TABLE = os.environ.get("RELAY_CONNECTIONS_TABLE", "RelayConnections")
SESSIONS_TABLE = os.environ.get("RELAY_SESSIONS_TABLE", "RelaySessions")
# Per-server secret discovery is intentionally pluggable — for MVP we
# read it from an SSM parameter named /imagine/relay/<server_id>/secret.
SSM_PREFIX = os.environ.get("RELAY_SECRET_SSM_PREFIX", "/imagine/relay")

_ddb = boto3.resource("dynamodb")
_ssm = boto3.client("ssm")


def _api_gateway_client(event):
    request = event.get("requestContext") or {}
    domain = request.get("domainName")
    stage = request.get("stage")
    endpoint = f"https://{domain}/{stage}"
    return boto3.client("apigatewaymanagementapi", endpoint_url=endpoint)


def _post(api, connection_id: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    api.post_to_connection(ConnectionId=connection_id, Data=data)


def _resolve_secret(server_id: str) -> str:
    parameter_name = f"{SSM_PREFIX}/{server_id}/secret"
    try:
        resp = _ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception:
        return ""


def _save_session(connection_id: str, server_id: str, role: str) -> None:
    sessions = _ddb.Table(SESSIONS_TABLE)
    now = int(time.time())
    sessions.put_item(
        Item={
            "server_id": server_id,
            "role": role,
            "connection_id": connection_id,
            "last_seen": now,
            "expires_at": now + RelayLimits.SESSION_MAX_DURATION_SECONDS,
        }
    )


def _lookup_server_connection(server_id: str) -> str:
    sessions = _ddb.Table(SESSIONS_TABLE)
    resp = sessions.get_item(Key={"server_id": server_id, "role": "server"})
    item = resp.get("Item")
    if not item:
        return ""
    return item.get("connection_id") or ""


def _known_servers() -> set[str]:
    """Set of currently-registered server_ids.

    The message handler uses this to authorize routing without trusting
    the originator's claim. For MVP we do a single Scan because the
    table size is bounded by the number of active Imagine servers.

    TODO(phase-9): replace with a paginated query against a dedicated
    'role=server' GSI once we have more than ~50 active servers, and
    add a cost guardrail so this Scan is never run on a hot path.
    """
    sessions = _ddb.Table(SESSIONS_TABLE)
    resp = sessions.scan(FilterExpression=Key("role").eq("server"))
    return {item["server_id"] for item in resp.get("Items", [])}


def _body_bytes(event) -> bytes:
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        return base64.b64decode(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    return bytes(body)


def handler(event, _context):
    request = event.get("requestContext") or {}
    connection_id = request.get("connectionId") or ""
    if not connection_id:
        return {"statusCode": 400, "body": "missing connectionId"}

    raw = _body_bytes(event)
    ok, err = validate_envelope_bytes(raw)
    api = _api_gateway_client(event)

    if not ok:
        _post(api, connection_id, make_error(err or "BAD_ENVELOPE"))
        return {"statusCode": 200, "body": "rejected"}

    envelope = json.loads(raw.decode("utf-8"))
    msg_type = envelope["type"]
    server_id = envelope["server_id"]
    msg_id = envelope.get("msg_id")

    if msg_type == "server.register":
        secret = _resolve_secret(server_id)
        ok, err = authorize_register(envelope, expected_secret=secret)
        if not ok:
            _post(api, connection_id, make_error(err, msg_id=msg_id))
            return {"statusCode": 200, "body": "auth_failed"}
        _save_session(connection_id, server_id, role="server")
        return {"statusCode": 200, "body": "registered"}

    if msg_type == "server.heartbeat":
        # We trust the connection mapping established at register time —
        # heartbeat just refreshes last_seen.
        sessions = _ddb.Table(SESSIONS_TABLE)
        sessions.update_item(
            Key={"server_id": server_id, "role": "server"},
            UpdateExpression="SET last_seen = :n",
            ExpressionAttributeValues={":n": int(time.time())},
        )
        return {"statusCode": 200, "body": "ok"}

    if msg_type in ("client.attach", "worker.attach"):
        kind = "client" if msg_type == "client.attach" else "worker"
        # TODO(phase-8): authorize_attach currently only checks the
        # token is non-empty. Phase 8 introduces per-server-issued
        # enrollment/attach tokens with revocation; once that lands,
        # validate the token against the server's public verification
        # key (sent at register time) instead of trusting any string.
        ok, err = authorize_attach(envelope, expected_kind=kind)
        if not ok:
            _post(api, connection_id, make_error(err, msg_id=msg_id))
            return {"statusCode": 200, "body": "auth_failed"}
        if server_id not in _known_servers():
            _post(api, connection_id, make_error("NO_SUCH_SERVER", msg_id=msg_id))
            return {"statusCode": 200, "body": "no_server"}
        _save_session(connection_id, server_id, role=kind)

    # Forward to the registered server's connection.
    # TODO(phase-7): server-to-worker responses need msg_id correlation
    # so the server can address a specific worker connection. For MVP we
    # only support originator → server fan-in; broadcasting back to a
    # specific worker requires storing (msg_id, requester_connection)
    # tuples here and routing on the response.
    decision = route(envelope, known_servers=_known_servers())
    if decision["action"] == "error":
        _post(
            api,
            connection_id,
            make_error(decision["code"], msg_id=decision.get("msg_id")),
        )
        return {"statusCode": 200, "body": "routed_error"}

    target_connection = _lookup_server_connection(decision["target"])
    if not target_connection:
        _post(api, connection_id, make_error("NO_SUCH_SERVER", msg_id=msg_id))
        return {"statusCode": 200, "body": "no_target"}

    _post(api, target_connection, envelope)
    return {"statusCode": 200, "body": "forwarded"}
