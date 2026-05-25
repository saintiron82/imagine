"""Lambda handler for the WebSocket $connect route.

This runs before any application-level auth. Its only job is to record
that a connection exists, with a short TTL — the actual identity is
checked on the first `server.register` / `client.attach` / `worker.attach`
message (see message.py).
"""

from __future__ import annotations

import json
import os
import time

import boto3

from auth import RelayLimits

CONNECTIONS_TABLE = os.environ.get("RELAY_CONNECTIONS_TABLE", "RelayConnections")

_ddb = boto3.resource("dynamodb")


def handler(event, _context):
    request = event.get("requestContext") or {}
    connection_id = request.get("connectionId")
    if not connection_id:
        return {"statusCode": 400, "body": "missing connectionId"}

    now = int(time.time())
    table = _ddb.Table(CONNECTIONS_TABLE)
    table.put_item(
        Item={
            "connection_id": connection_id,
            "connected_at": now,
            # TTL — DynamoDB sweeps this row once the session max
            # duration is reached, even if the disconnect handler is
            # never invoked (clients can vanish mid-route).
            "expires_at": now + RelayLimits.SESSION_MAX_DURATION_SECONDS,
        }
    )
    return {"statusCode": 200, "body": json.dumps({"ok": True})}
