"""
Phase 6A: E-commerce Recommendation Service

Thin application layer around the existing recommendation pipeline.

This module does not retrain or replace the recommendation models.
It loads the existing trained artifacts and exposes recommendations
for an individual user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import pandas as pd

from src.recommendation_pipeline import topk_popularity, topk_svd


class RecommendationService:
    """Serve recommendations using existing trained models."""

    def __init__(
        self,
        model_path: str = "models/recommendation_models.joblib",
        metadata_path: str = "models/training_metadata.json",
        data_path: str = "data/processed/events_cleaned.csv",
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.data_path = Path(data_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Recommendation model not found: {self.model_path}"
            )

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Interaction data not found: {self.data_path}"
            )

        self.models = joblib.load(self.model_path)

        self.metadata: Dict[str, Any] = {}

        if self.metadata_path.exists():
            with self.metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                self.metadata = json.load(file)

        self.events = pd.read_csv(
            self.data_path,
            usecols=["visitorid", "itemid", "event"],
        )

        self.events["visitorid"] = self.events["visitorid"].astype(int)
        self.events["itemid"] = self.events["itemid"].astype(int)

    def recommend(
        self,
        visitor_id: int,
        method: str = "svd",
        k: int = 10,
    ) -> List[int]:
        """
        Return top-k recommendations for one visitor.

        Existing users are served by the trained model.
        Unknown users fall back to popularity.
        """

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        method = method.lower()

        if method not in {"svd", "popularity"}:
            raise ValueError(
                "method must be either 'svd' or 'popularity'."
            )

        user_history = self.events[
            self.events["visitorid"] == visitor_id
        ]

        if user_history.empty:
            return self._popularity_fallback(
                visitor_id=visitor_id,
                k=k,
            )

        if method == "popularity":
            return self._popularity_fallback(
                visitor_id=visitor_id,
                k=k,
            )

        return self._svd_recommendations(
            visitor_id=visitor_id,
            k=k,
        )

    def _popularity_fallback(
        self,
        visitor_id: int,
        k: int,
    ) -> List[int]:
        """Return popularity recommendations."""

        history = set(
            self.events.loc[
                self.events["visitorid"] == visitor_id,
                "itemid",
            ].astype(int)
        )

        popularity = (
            self.events
            .groupby("itemid")
            .size()
            .sort_values(ascending=False)
        )

        recommendations = [
            int(item_id)
            for item_id in popularity.index
            if int(item_id) not in history
        ]

        return recommendations[:k]

    def _svd_recommendations(
        self,
        visitor_id: int,
        k: int,
    ) -> List[int]:
        """Generate SVD recommendations using the existing model."""

        try:
            recommendations = topk_svd(
                model=self.models,
                user_id=visitor_id,
                k=k,
            )
        except (KeyError, ValueError, TypeError):
            return self._popularity_fallback(
                visitor_id=visitor_id,
                k=k,
            )

        return [int(item_id) for item_id in recommendations]


def get_recommendations(
    visitor_id: int,
    method: str = "svd",
    k: int = 10,
) -> List[int]:
    """Convenience function for application code."""

    service = RecommendationService()

    return service.recommend(
        visitor_id=visitor_id,
        method=method,
        k=k,
    )
