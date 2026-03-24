"""WebSocket broadcast hub for real-time dashboard updates."""

import asyncio
import json
from typing import Any, Dict, Set

import websockets
from websockets.server import WebSocketServerProtocol

from utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketHub:
    """Broadcast pipeline events and metrics to all connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: Set[WebSocketServerProtocol] = set()

    async def handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle a new WebSocket connection."""
        self._clients.add(websocket)
        logger.info("WebSocketHub: client connected (total=%d).", len(self._clients))
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            logger.info(
                "WebSocketHub: client disconnected (total=%d).", len(self._clients)
            )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
        if not self._clients:
            return
        payload = json.dumps(message)
        await asyncio.gather(
            *[self._safe_send(client, payload) for client in list(self._clients)],
            return_exceptions=True,
        )

    async def _safe_send(self, client: WebSocketServerProtocol, payload: str) -> None:
        try:
            await client.send(payload)
        except Exception as exc:
            logger.debug("WebSocketHub: send failed: %s", exc)
            self._clients.discard(client)

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """Start the WebSocket server."""
        logger.info("WebSocketHub: starting on ws://%s:%d", host, port)
        async with websockets.serve(self.handler, host, port):
            await asyncio.Future()  # run forever


# Module-level singleton
_hub = WebSocketHub()


def get_hub() -> WebSocketHub:
    return _hub
