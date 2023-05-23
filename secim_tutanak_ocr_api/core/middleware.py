from fastapi import FastAPI
from starlette.types import ASGIApp, Scope, Receive, Send


class SimpleASGIMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        await self.app(scope, receive, send)
        client = scope["client"]
        print(f"[CLIENT]: {client}")
        print(send)