from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
import sys
from typing import Any

from .execution import Q4ExecutionRequest, Q4ExecutionResponse
from .models import FailureCategory, Q4ArtifactMode, Q4ExecutionStatus

LABEL_COLUMN = "Consumer disputed?"
IDENTIFIER_COLUMN = "Complaint ID"


class _WorkerFailure(Exception):
    def __init__(
        self,
        category: FailureCategory,
        reason: str,
        *,
        logs: list[str] | None = None,
    ) -> None:
        super().__init__(reason)
        self.category = category
        self.reason = reason
        self.logs = list(logs or [])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m ml26_grader.q4.worker <request_path> <response_path>", file=sys.stderr)
        return 2

    request_path = Path(args[0])
    response_path = Path(args[1])

    try:
        request = Q4ExecutionRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
        response = _run_request(request)
    except _WorkerFailure as exc:
        response = Q4ExecutionResponse(
            backend_name="subprocess",
            execution_status=Q4ExecutionStatus.FAILED,
            failure_category=exc.category,
            failure_reason=exc.reason,
            execution_logs=exc.logs,
        )
    except Exception as exc:  # pragma: no cover - last-resort worker guard
        response = Q4ExecutionResponse(
            backend_name="subprocess",
            execution_status=Q4ExecutionStatus.FAILED,
            failure_category=FailureCategory.INFERENCE_FAILURE,
            failure_reason=f"Unexpected Q4 worker failure: {exc}",
            execution_logs=["The Q4 worker raised an unexpected unclassified exception."],
        )

    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


def _run_request(request: Q4ExecutionRequest) -> Q4ExecutionResponse:
    logs: list[str] = []
    pd = _import_pandas(logs)
    dataset_frame = _load_dataset_frame(pd, request.dataset_path, logs)
    feature_frame = dataset_frame.drop(
        columns=[LABEL_COLUMN, IDENTIFIER_COLUMN],
        errors="ignore",
    )
    logs.append(
        f"Prepared feature frame with {len(feature_frame)} rows and {len(feature_frame.columns)} columns."
    )

    if request.artifact_layout.feature_engineering_required:
        feature_engineering_file = request.artifact_layout.feature_engineering_file
        if feature_engineering_file is None:
            raise _WorkerFailure(
                FailureCategory.IMPORT_FAILURE,
                "feature_engineering.py was required but no file path was provided.",
                logs=logs,
            )
        _load_feature_engineering_module(feature_engineering_file, logs)

    predictions: list[Any]
    if request.artifact_layout.artifact_mode == Q4ArtifactMode.COMBINED_PIPELINE:
        combined_pipeline = _load_pickle(
            request.artifact_layout.combined_pipeline,
            "combined pipeline",
            logs,
        )
        try:
            raw_predictions = combined_pipeline.predict(feature_frame)
        except Exception as exc:
            raise _WorkerFailure(
                FailureCategory.INFERENCE_FAILURE,
                f"Combined pipeline prediction failed: {exc}",
                logs=logs,
            ) from exc
        predictions = _coerce_predictions(raw_predictions)
    elif request.artifact_layout.artifact_mode == Q4ArtifactMode.SPLIT_PIPELINE:
        preprocessor = _load_pickle(
            request.artifact_layout.split_preprocessor,
            "split preprocessor",
            logs,
        )
        model = _load_pickle(
            request.artifact_layout.split_model,
            "split model",
            logs,
        )
        transformed = _apply_preprocessor(preprocessor, feature_frame, logs)
        predictions = _predict_with_split_model(pd, model, transformed, logs)
    else:
        raise _WorkerFailure(
            FailureCategory.LOAD_FAILURE,
            f"Unsupported Q4 artifact mode: {request.artifact_layout.artifact_mode.value}",
            logs=logs,
        )

    logs.append(f"Generated {len(predictions)} predictions from student artifacts.")
    return Q4ExecutionResponse(
        backend_name="subprocess",
        execution_status=Q4ExecutionStatus.SUCCEEDED,
        predictions=predictions,
        execution_logs=logs,
    )


def _import_pandas(logs: list[str]):  # type: ignore[no-untyped-def]
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise _WorkerFailure(
            FailureCategory.IMPORT_FAILURE,
            f"Failed to import pandas for Q4 execution: {exc}",
            logs=logs,
        ) from exc

    logs.append(f"Imported pandas {pd.__version__} for Q4 execution.")
    return pd


def _load_dataset_frame(pd, dataset_path: Path, logs: list[str]):  # type: ignore[no-untyped-def]
    resolved_dataset_path = dataset_path.resolve()
    try:
        dataset_frame = pd.read_csv(resolved_dataset_path, encoding="utf-8-sig")
    except Exception as exc:
        raise _WorkerFailure(
            FailureCategory.LOAD_FAILURE,
            f"Failed to load evaluation dataset {resolved_dataset_path}: {exc}",
            logs=logs,
        ) from exc

    logs.append(f"Loaded evaluation dataset from {resolved_dataset_path}.")
    logs.append(f"Dataset frame shape: {dataset_frame.shape[0]} rows x {dataset_frame.shape[1]} columns.")
    return dataset_frame


def _load_feature_engineering_module(path: Path, logs: list[str]) -> None:
    try:
        spec = importlib.util.spec_from_file_location("feature_engineering", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build an import spec for {path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules["feature_engineering"] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise _WorkerFailure(
            FailureCategory.IMPORT_FAILURE,
            f"Failed to import feature_engineering.py from {path}: {exc}",
            logs=logs,
        ) from exc

    logs.append(f"Imported feature_engineering.py from {path}.")


def _load_pickle(path: Path | None, label: str, logs: list[str]) -> Any:
    if path is None:
        raise _WorkerFailure(
            FailureCategory.IMPORT_FAILURE,
            f"Missing {label} artifact path.",
            logs=logs,
        )
    try:
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
    except Exception as exc:
        raise _WorkerFailure(
            FailureCategory.IMPORT_FAILURE,
            f"Failed to load {label} artifact from {path}: {exc}",
            logs=logs,
        ) from exc

    logs.append(f"Loaded {label} artifact from {path}.")
    return artifact


def _apply_preprocessor(preprocessor: Any, feature_frame: Any, logs: list[str]) -> Any:
    try:
        if callable(preprocessor):
            logs.append("Applying callable split preprocessor.")
            return preprocessor(feature_frame.copy())
        if hasattr(preprocessor, "transform"):
            logs.append("Applying split preprocessor via .transform(...).")
            return preprocessor.transform(feature_frame)
    except Exception as exc:
        raise _WorkerFailure(
            FailureCategory.INFERENCE_FAILURE,
            f"Split preprocessor execution failed: {exc}",
            logs=logs,
        ) from exc

    raise _WorkerFailure(
        FailureCategory.IMPORT_FAILURE,
        "Split preprocessor artifact is neither callable nor transformable.",
        logs=logs,
    )


def _predict_with_split_model(pd, model: Any, transformed: Any, logs: list[str]) -> list[Any]:  # type: ignore[no-untyped-def]
    try:
        raw_predictions = model.predict(transformed)
    except Exception as exc:
        if isinstance(transformed, pd.DataFrame):
            logs.append(
                "Split model prediction failed on the original DataFrame; retrying with string-cast values."
            )
            try:
                raw_predictions = model.predict(transformed.astype(str))
            except Exception as retry_exc:
                raise _WorkerFailure(
                    FailureCategory.INFERENCE_FAILURE,
                    f"Split model prediction failed after string-cast retry: {retry_exc}",
                    logs=logs,
                ) from retry_exc
        else:
            raise _WorkerFailure(
                FailureCategory.INFERENCE_FAILURE,
                f"Split model prediction failed: {exc}",
                logs=logs,
            ) from exc

    return _coerce_predictions(raw_predictions)


def _coerce_predictions(raw_predictions: Any) -> list[Any]:
    if hasattr(raw_predictions, "tolist"):
        converted = raw_predictions.tolist()
    elif isinstance(raw_predictions, list):
        converted = raw_predictions
    elif isinstance(raw_predictions, tuple):
        converted = list(raw_predictions)
    elif isinstance(raw_predictions, (str, bytes)):
        converted = [raw_predictions]
    else:
        try:
            converted = list(raw_predictions)
        except TypeError:
            converted = [raw_predictions]

    if isinstance(converted, list):
        return converted
    if isinstance(converted, tuple):
        return list(converted)
    if isinstance(converted, (str, bytes)):
        return [converted]
    try:
        return list(converted)
    except TypeError:
        return [converted]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
