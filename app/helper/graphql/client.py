import json
from datetime import date, datetime
from typing import Any
from uuid import UUID

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

ENDPOINT = ""
ADMIN_SECRET = ""


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def json_serializer(obj: Any):
    return json.dumps(obj, cls=CustomEncoder)


class GraphQLClient:
    def __init__(self, url, headers):
        self._client = Client(
            transport=AIOHTTPTransport(
                url=url, headers=headers, json_serialize=json_serializer
            )
        )
        self._session = None

    def load_query(self, path):
        with open(path) as f:
            return gql(f.read())

    async def connect(self):
        self._session = await self._client.connect_async(reconnecting=True)

    async def close(self):
        await self._client.close_async()

    async def execute(self, path, variable_values):
        query = self.load_query(path)
        return await self._session.execute(query, variable_values=variable_values)


async def get_gql_client():
    try:
        client = GraphQLClient(
            url=ENDPOINT,
            headers={"x-hasura-admin-secret": ADMIN_SECRET},
        )
        await client.connect()
        yield client
    finally:
        await client.close()
