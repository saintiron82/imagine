"""Lambda handler for the WebSocket $disconnect route.

We remove the connection row plus any session rows that referenced it.
DynamoDB TTL still acts as a safety net, but explicit cleanup keeps the
RelaySessions table small enough to scan in the message handler.
"""

from __future__ import annotations

import json
import os

import boto3
from boto3.dynamodb.conditions import Key

CONNECTIONS_TABLE = os.environ.get("RELAY_CONNECTIONS_TABLE", "RelayConnections")
SESSIONS_TABLE = os.environ.get("RELAY_SESSIONS_TABLE", "RelaySessions")

_ddb = boto3.resource("dynamodb")


def handler(event, _context):
    request = event.get("requestContext") or {}
    connection_id = request.get("connectionId")
    if not connection_id:
        return {"statusCode": 400, "body": "missing connectionId"}

    conn_table = _ddb.Table(CONNECTIONS_TABLE)
    conn_table.delete_item(Key={"connection_id": connection_id})

    sessions = _ddb.Table(SESSIONS_TABLE)
    # ConnectionIndex projects (connection_id, server_id, role). We use
    # it to find the matching session row(s) on disconnect.
    resp = sessions.query(
        IndexName="ConnectionIndex",
        KeyConditionExpression=Key("connection_id").eq(connection_id),
    )
    for item in resp.get("Items", []):
        sessions.delete_item(
            Key={
                "server_id": item["server_id"],
                "role": item["role"],
            }
        )

    return {"statusCode": 200, "body": json.dumps({"ok": True})}
