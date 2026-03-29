from __future__ import annotations


class ThresholdPredictor:
    def __init__(self, column: str, threshold: float) -> None:
        self.column = column
        self.threshold = threshold

    def predict(self, frame):  # type: ignore[no-untyped-def]
        return [1 if float(value) >= self.threshold else 0 for value in frame[self.column]]


class StringThresholdPredictor:
    def __init__(self, column: str, threshold: float) -> None:
        self.column = column
        self.threshold = threshold

    def predict(self, frame):  # type: ignore[no-untyped-def]
        values = list(frame[self.column])
        if any(not isinstance(value, str) for value in values):
            raise ValueError("Expected string-valued features.")
        return [1 if float(value) >= self.threshold else 0 for value in values]


class FixedPredictionPipeline:
    def __init__(self, predictions: list[int | float | str | bool]) -> None:
        self.predictions = list(predictions)

    def predict(self, frame):  # type: ignore[no-untyped-def]
        return list(self.predictions)


class RaisingPredictor:
    def __init__(self, message: str) -> None:
        self.message = message

    def predict(self, frame):  # type: ignore[no-untyped-def]
        raise RuntimeError(self.message)


class PassthroughTransformer:
    def transform(self, frame):  # type: ignore[no-untyped-def]
        return frame.copy()
