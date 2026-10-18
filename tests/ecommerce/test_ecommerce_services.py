from pathlib import Path

import pytest

from src.ecommerce.event_logger import EventLogger
from src.ecommerce.recommendation_service import RecommendationService


def test_event_logger_records_event(tmp_path: Path):
    output = tmp_path / "events.csv"

    logger = EventLogger(
        output_path=str(output)
    )

    record = logger.log_event(
        visitor_id=7,
        item_id=369447,
        event="view",
    )

    assert record["visitorid"] == 7
    assert record["itemid"] == 369447
    assert record["event"] == "view"

    events = logger.read_events()

    assert len(events) == 1
    assert events.iloc[0]["visitorid"] == 7
    assert events.iloc[0]["itemid"] == 369447
    assert events.iloc[0]["event"] == "view"


def test_event_logger_rejects_invalid_event(tmp_path: Path):
    logger = EventLogger(
        output_path=str(tmp_path / "events.csv")
    )

    with pytest.raises(ValueError):
        logger.log_event(
            visitor_id=7,
            item_id=369447,
            event="invalid_event",
        )


def test_recommendation_service_known_user():
    service = RecommendationService()

    recommendations = service.recommend(
        visitor_id=7,
        method="svd",
        k=10,
    )

    assert len(recommendations) == 10
    assert all(
        isinstance(item_id, int)
        for item_id in recommendations
    )


def test_recommendation_service_unknown_user():
    service = RecommendationService()

    recommendations = service.recommend(
        visitor_id=999999999,
        method="svd",
        k=10,
    )

    assert len(recommendations) == 10
    assert all(
        isinstance(item_id, int)
        for item_id in recommendations
    )


def test_recommendation_service_rejects_invalid_method():
    service = RecommendationService()

    with pytest.raises(ValueError):
        service.recommend(
            visitor_id=7,
            method="invalid",
            k=10,
        )
