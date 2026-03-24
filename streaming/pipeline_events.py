"""Pipeline stage events: publish stage transitions to subscribers."""

import asyncio
import json
from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class PipelineEvents:
    """Publish/subscribe bus for pipeline stage events."""

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        """Return a new async queue that will receive all future events."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def publish(self, stage: str, data: Dict[str, Any]) -> None:
        """Publish a stage event to all subscribers."""
        event = json.dumps({"stage": stage, **data})
        for q in list(self._subscribers):
            try:
                await q.put(event)
            except Exception as exc:
                logger.warning(
                    "PipelineEvents: failed to publish to subscriber: %s", exc
                )

    async def publish_stage_start(self, stage: str, query_id: str = "") -> None:
        await self.publish(stage, {"status": "start", "query_id": query_id})

    async def publish_stage_end(
        self, stage: str, query_id: str = "", duration_ms: float = 0.0
    ) -> None:
        await self.publish(
            stage, {"status": "end", "query_id": query_id, "duration_ms": duration_ms}
        )


# Module-level singleton
_events = PipelineEvents()


def get_events() -> PipelineEvents:
    return _events
