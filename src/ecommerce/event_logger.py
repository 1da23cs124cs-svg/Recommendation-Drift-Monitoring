"""
Phase 6B: E-commerce Event Logger

Records application interactions separately from the historical
research dataset.

Supported events:
    view
    click
    addtocart
    transaction
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


VALID_EVENTS = {
    "view",
    "click",
    "addtocart",
    "transaction",
}

EVENT_COLUMNS = [
    "timestamp",
    "visitorid",
    "itemid",
    "event",
]


class EventLogger:
    """Append e-commerce interaction events to a CSV file."""

    def __init__(
        self,
        output_path: str = "data/processed/ecommerce_events.csv",
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.output_path.exists():
            pd.DataFrame(
                columns=EVENT_COLUMNS
            ).to_csv(
                self.output_path,
                index=False,
            )

    def log_event(
        self,
        visitor_id: int,
        item_id: int,
        event: str,
    ) -> dict:
        """Record one interaction event."""

        event = event.lower().strip()

        if event not in VALID_EVENTS:
            raise ValueError(
                f"Unsupported event '{event}'. "
                f"Expected one of: {sorted(VALID_EVENTS)}"
            )

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "visitorid": int(visitor_id),
            "itemid": int(item_id),
            "event": event,
        }

        pd.DataFrame(
            [record],
            columns=EVENT_COLUMNS,
        ).to_csv(
            self.output_path,
            mode="a",
            header=False,
            index=False,
        )

        return record

    def read_events(self) -> pd.DataFrame:
        """Read all application events."""

        return pd.read_csv(
            self.output_path
        )
