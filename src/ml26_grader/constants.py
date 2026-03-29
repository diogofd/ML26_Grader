from __future__ import annotations

from typing import Final

CONFIDENCE_REVIEW_THRESHOLD: Final[float] = 8.5

QUESTION_SUBQUESTION_IDS: Final[dict[str, tuple[str, str]]] = {
    "Q2": ("Q2.1", "Q2.2"),
    "Q3": ("Q3.1", "Q3.2"),
}

PUBLIC_DATASET_COLUMNS: Final[tuple[str, ...]] = (
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
)
