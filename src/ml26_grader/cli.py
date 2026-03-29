from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
from typing import Sequence

from pydantic import ValidationError

from .config import GradingConfig
from .extraction.service import NotebookEvidenceExtractor
from .ingest.batch import (
    BatchDiscoveryEntry,
    BatchExtractionResult,
    discover_batch_submissions,
    extract_submission_zip,
)
from .ingest.datasets import dataset_manifest_map
from .q4.deterministic import assign_leaderboard_ranks
from .q4.pipeline import Q4EvaluationPipeline
from .q4.models import (
    FailureCategory,
    LeaderboardEntry,
    LeaderboardStatus,
    Q4ArtifactLayout,
    Q4EvaluationResult,
    Q4ExecutionStatus,
)
from .reporting.render import render_json_document
from .scoring.models import QuestionGradingResult, QuestionGradingStatus
from .scoring.pipeline import Q23GradingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml26-grader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config = subparsers.add_parser("validate-config", help="Validate the grader TOML config.")
    validate_config.add_argument("config", type=Path)

    list_datasets = subparsers.add_parser(
        "list-public-datasets",
        help="Show the public dataset manifests baked into the scaffold.",
    )
    list_datasets.add_argument("--base-dir", type=Path, default=Path("."))

    extract = subparsers.add_parser(
        "extract-evidence",
        help="Run the notebook evidence extractor for Q2 or Q3.",
    )
    extract.add_argument("submission", type=Path)
    extract.add_argument("question_id", choices=("Q2", "Q3"))

    inspect_q4 = subparsers.add_parser(
        "inspect-q4",
        help="Run Q4 artifact validation, execution, and deterministic result checks.",
    )
    inspect_q4.add_argument("submission", type=Path)
    inspect_q4.add_argument(
        "--config",
        type=Path,
        default=Path("config/grading.toml"),
    )
    inspect_q4.add_argument(
        "--dataset",
        choices=("training", "test", "modeltesting"),
        default="modeltesting",
    )

    grade_q23 = subparsers.add_parser(
        "grade-q23-submission",
        help="Grade one extracted submission directory for Q2 and Q3.",
    )
    grade_q23.add_argument("submission", type=Path)
    grade_q23.add_argument(
        "--config",
        type=Path,
        default=Path("config/grading.toml"),
    )
    grade_q23.add_argument(
        "--questions",
        type=Path,
        default=Path("specs/questions.toml"),
    )
    grade_q23.add_argument(
        "--rubrics",
        type=Path,
        default=Path("specs/rubrics.toml"),
    )
    grade_q23.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    grade_q23_batch = subparsers.add_parser(
        "grade-q23-batch",
        help="Discover zip submissions in a batch directory, extract them safely, and grade Q2/Q3.",
    )
    grade_q23_batch.add_argument("batch_dir", type=Path)
    grade_q23_batch.add_argument(
        "--config",
        type=Path,
        default=Path("config/grading.toml"),
    )
    grade_q23_batch.add_argument(
        "--questions",
        type=Path,
        default=Path("specs/questions.toml"),
    )
    grade_q23_batch.add_argument(
        "--rubrics",
        type=Path,
        default=Path("specs/rubrics.toml"),
    )
    grade_q23_batch.add_argument(
        "--extract-root",
        type=Path,
        default=Path("sandbox/extracted"),
    )
    grade_q23_batch.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/batch_q23"),
    )

    grade_q4_batch = subparsers.add_parser(
        "grade-q4-batch",
        help="Discover or reuse extracted submissions in a batch directory and evaluate Q4.",
    )
    grade_q4_batch.add_argument("batch_dir", type=Path)
    grade_q4_batch.add_argument(
        "--config",
        type=Path,
        default=Path("config/grading.toml"),
    )
    grade_q4_batch.add_argument(
        "--extract-root",
        type=Path,
        default=Path("sandbox/extracted"),
    )
    grade_q4_batch.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/batch_q4"),
    )
    grade_q4_batch.add_argument(
        "--dataset",
        choices=("training", "test", "modeltesting"),
        default="modeltesting",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        config = GradingConfig.from_toml(args.config)
        print(render_json_document(config))
        return 0

    if args.command == "list-public-datasets":
        manifests = dataset_manifest_map(args.base_dir)
        print(render_json_document(manifests))
        return 0

    if args.command == "extract-evidence":
        extractor = NotebookEvidenceExtractor()
        result = extractor.extract(args.submission, args.question_id)
        print(render_json_document(result))
        return 0

    if args.command == "inspect-q4":
        result, exit_code = inspect_single_submission_q4(
            submission_dir=args.submission,
            config_path=args.config,
            dataset_name=args.dataset,
        )
        print(render_json_document(result))
        return exit_code

    if args.command == "grade-q23-submission":
        payload, exit_code = grade_single_submission_q23(
            submission_dir=args.submission,
            config_path=args.config,
            questions_path=args.questions,
            rubrics_path=args.rubrics,
        )
        rendered = render_json_document(payload)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return exit_code

    if args.command == "grade-q23-batch":
        payload, exit_code = grade_batch_q23(
            batch_dir=args.batch_dir,
            config_path=args.config,
            questions_path=args.questions,
            rubrics_path=args.rubrics,
            extract_root=args.extract_root,
            output_dir=args.output_dir,
        )
        print(render_json_document(payload))
        return exit_code

    if args.command == "grade-q4-batch":
        payload, exit_code = grade_batch_q4(
            batch_dir=args.batch_dir,
            config_path=args.config,
            extract_root=args.extract_root,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
        )
        print(render_json_document(payload))
        return exit_code

    parser.error(f"Unsupported command: {args.command}")
    return 2


def grade_single_submission_q23(
    *,
    submission_dir: Path,
    config_path: Path,
    questions_path: Path,
    rubrics_path: Path,
) -> tuple[dict[str, Any], int]:
    resolved_submission = submission_dir.resolve()
    base_payload = {
        "submission_dir": resolved_submission,
        "config_path": config_path.resolve(),
        "questions_path": questions_path.resolve(),
        "rubrics_path": rubrics_path.resolve(),
    }

    if not resolved_submission.exists() or not resolved_submission.is_dir():
        return (
            {
                **base_payload,
                "status": "failed",
                "review_required": True,
                "review_reasons": ["submission_directory_missing"],
                "failure_reason": f"{resolved_submission} is not an extracted submission directory.",
                "results": {},
            },
            1,
        )

    try:
        config = GradingConfig.from_toml(config_path)
    except (FileNotFoundError, OSError, ValidationError, ValueError) as exc:
        return (
            {
                **base_payload,
                "status": "failed",
                "review_required": True,
                "review_reasons": ["config_load_failed"],
                "failure_reason": f"Config loading failed: {exc}",
                "results": {},
            },
            1,
        )

    pipeline = Q23GradingPipeline.from_paths(
        config,
        questions_path,
        rubrics_path,
        extractor=NotebookEvidenceExtractor(config.submission),
    )
    results = {
        question_id: pipeline.grade_question(
            resolved_submission,
            question_id,
            metadata={"submission_dir": str(resolved_submission)},
        )
        for question_id in ("Q2", "Q3")
    }
    overall_status = _overall_q23_status(results)
    overall_review_reasons = _merged_review_reasons(results)

    payload = {
        **base_payload,
        "status": overall_status,
        "review_required": any(result.review_required for result in results.values()),
        "review_reasons": overall_review_reasons,
        "results": results,
    }
    if overall_status == "failed":
        payload["failure_reason"] = "One or more questions failed closed during Q2/Q3 grading."
        return payload, 1
    return payload, 0


def inspect_single_submission_q4(
    *,
    submission_dir: Path,
    config_path: Path,
    dataset_name: str = "modeltesting",
) -> tuple[Q4EvaluationResult, int]:
    resolved_submission = submission_dir.resolve()

    try:
        config = GradingConfig.from_toml(config_path)
    except (FileNotFoundError, OSError, ValidationError, ValueError) as exc:
        return (
            Q4EvaluationResult(
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                artifact_layout=Q4ArtifactLayout(
                    submission_root=resolved_submission,
                    missing_artifacts=["config"],
                ),
                predictions_valid=False,
                labels_available=False,
                failure_category=FailureCategory.LOAD_FAILURE,
                failure_reason=f"Config loading failed: {exc}",
                execution_logs=["Q4 inspection failed before execution because config loading failed."],
            ),
            1,
        )

    pipeline = Q4EvaluationPipeline.from_config(config)
    result = pipeline.evaluate(resolved_submission, dataset_name=dataset_name)
    return result, 0 if result.execution_status == Q4ExecutionStatus.SUCCEEDED else 1


def grade_batch_q23(
    *,
    batch_dir: Path,
    config_path: Path,
    questions_path: Path,
    rubrics_path: Path,
    extract_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], int]:
    resolved_batch_dir = batch_dir.resolve()
    resolved_extract_root = extract_root.resolve()
    resolved_output_dir = output_dir.resolve()
    submissions_output_dir = resolved_output_dir / "submissions"

    base_payload = {
        "batch_dir": resolved_batch_dir,
        "config_path": config_path.resolve(),
        "questions_path": questions_path.resolve(),
        "rubrics_path": rubrics_path.resolve(),
        "extract_root": resolved_extract_root,
        "output_dir": resolved_output_dir,
    }

    if not resolved_batch_dir.exists() or not resolved_batch_dir.is_dir():
        return (
            {
                **base_payload,
                "status": "failed",
                "failure_reason": f"{resolved_batch_dir} is not a batch directory.",
                "review_required": True,
                "review_reasons": ["batch_directory_missing"],
                "manifest": None,
                "submissions": [],
                "summary_csv_path": None,
                "summary_json_path": None,
            },
            1,
        )

    submissions_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = discover_batch_submissions(resolved_batch_dir)
    submission_payloads: list[dict[str, Any]] = []
    summary_rows: list[dict[str, str]] = []

    for entry in manifest.submissions:
        extraction_result = extract_submission_zip(entry, resolved_extract_root)
        if extraction_result.status == "ready" and extraction_result.extracted_submission_path is not None:
            grading_payload, grading_exit_code = grade_single_submission_q23(
                submission_dir=extraction_result.extracted_submission_path,
                config_path=config_path,
                questions_path=questions_path,
                rubrics_path=rubrics_path,
            )
            per_submission_payload = {
                "student_folder_name": entry.student_folder_name,
                "student_folder_path": entry.student_folder_path,
                "zip_path": entry.selected_zip_path,
                "discovery_warnings": entry.warnings,
                "extraction": extraction_result,
                "grading": grading_payload,
            }
        else:
            grading_exit_code = 1
            grading_payload = {
                "status": "failed",
                "review_required": True,
                "review_reasons": [warning.code for warning in entry.warnings],
                "failure_reason": extraction_result.failure_reason,
                "results": {},
            }
            per_submission_payload = {
                "student_folder_name": entry.student_folder_name,
                "student_folder_path": entry.student_folder_path,
                "zip_path": entry.selected_zip_path,
                "discovery_warnings": entry.warnings,
                "extraction": extraction_result,
                "grading": grading_payload,
            }

        result_output_path = submissions_output_dir / f"{_slugify(entry.student_folder_name)}.json"
        result_output_path.write_text(
            render_json_document(per_submission_payload) + "\n",
            encoding="utf-8",
        )
        per_submission_payload["result_json_path"] = result_output_path
        submission_payloads.append(per_submission_payload)
        summary_rows.append(
            {
                "student_folder_name": entry.student_folder_name,
                "student_folder_path": str(entry.student_folder_path),
                "zip_path": str(entry.selected_zip_path) if entry.selected_zip_path is not None else "",
                "extracted_submission_path": (
                    str(extraction_result.extracted_submission_path)
                    if extraction_result.extracted_submission_path is not None
                    else ""
                ),
                "grading_status": str(grading_payload.get("status", "failed")),
                "review_required": str(bool(grading_payload.get("review_required", True))).lower(),
                "review_reasons": "|".join(str(reason) for reason in grading_payload.get("review_reasons", [])),
                "failure_reason": str(grading_payload.get("failure_reason") or extraction_result.failure_reason or ""),
                "result_json_path": str(result_output_path),
            }
        )
        per_submission_payload["exit_code"] = grading_exit_code

    summary_json_path = resolved_output_dir / "batch_summary.json"
    summary_csv_path = resolved_output_dir / "batch_summary.csv"
    aggregate_payload = {
        **base_payload,
        "status": _overall_batch_status(submission_payloads),
        "review_required": any(
            bool(submission["grading"].get("review_required", True))
            for submission in submission_payloads
        ),
        "review_reasons": _merge_batch_review_reasons(submission_payloads),
        "manifest": manifest,
        "submissions": submission_payloads,
        "summary_json_path": summary_json_path,
        "summary_csv_path": summary_csv_path,
    }

    summary_json_path.write_text(
        render_json_document(aggregate_payload) + "\n",
        encoding="utf-8",
    )
    _write_batch_summary_csv(summary_csv_path, summary_rows)

    exit_code = 1 if aggregate_payload["status"] == "failed" else 0
    return aggregate_payload, exit_code


def grade_batch_q4(
    *,
    batch_dir: Path,
    config_path: Path,
    extract_root: Path,
    output_dir: Path,
    dataset_name: str,
) -> tuple[dict[str, Any], int]:
    resolved_batch_dir = batch_dir.resolve()
    resolved_extract_root = extract_root.resolve()
    resolved_output_dir = output_dir.resolve()
    submissions_output_dir = resolved_output_dir / "submissions"

    base_payload = {
        "batch_dir": resolved_batch_dir,
        "config_path": config_path.resolve(),
        "extract_root": resolved_extract_root,
        "output_dir": resolved_output_dir,
        "dataset_name": dataset_name,
    }

    if not resolved_batch_dir.exists() or not resolved_batch_dir.is_dir():
        return (
            {
                **base_payload,
                "status": "failed",
                "failure_reason": f"{resolved_batch_dir} is not a batch directory.",
                "manifest": None,
                "submissions": [],
                "summary_json_path": None,
                "summary_csv_path": None,
                "leaderboard_csv_path": None,
                "leaderboard": [],
            },
            1,
        )

    try:
        config = GradingConfig.from_toml(config_path)
    except (FileNotFoundError, OSError, ValidationError, ValueError) as exc:
        return (
            {
                **base_payload,
                "status": "failed",
                "failure_reason": f"Config loading failed: {exc}",
                "manifest": None,
                "submissions": [],
                "summary_json_path": None,
                "summary_csv_path": None,
                "leaderboard_csv_path": None,
                "leaderboard": [],
            },
            1,
        )

    submissions_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = discover_batch_submissions(resolved_batch_dir)
    pipeline = Q4EvaluationPipeline.from_config(config)
    submission_payloads: list[dict[str, Any]] = []

    for entry in manifest.submissions:
        extraction_result = _prepare_batch_q4_submission(entry, resolved_extract_root)
        if extraction_result.status == "ready" and extraction_result.extracted_submission_path is not None:
            q4_result = pipeline.evaluate(
                extraction_result.extracted_submission_path,
                dataset_name=dataset_name,
            )
        else:
            base_failure_reason = (
                extraction_result.failure_reason
                or "Submission could not be prepared for Q4 execution."
            )
            q4_result = Q4EvaluationResult(
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                artifact_layout=Q4ArtifactLayout(
                    submission_root=entry.student_folder_path.resolve(),
                    missing_artifacts=["submission_root"],
                ),
                predictions_valid=False,
                labels_available=False,
                zero_grade_policy_applied=True,
                zero_grade_policy_reason="non_functional_model",
                failure_category=FailureCategory.MISSING_ARTIFACTS,
                failure_reason=(
                    f"{base_failure_reason} "
                    "Q4 is zero by policy because the submission did not successfully produce valid binary predictions."
                ),
                execution_logs=[
                    "Q4 is zero by policy because the submission could not be prepared or did not produce valid binary predictions.",
                ],
            )

        submission_payloads.append(
            {
                "student_folder_name": entry.student_folder_name,
                "student_folder_path": entry.student_folder_path,
                "zip_path": entry.selected_zip_path,
                "discovery_warnings": entry.warnings,
                "extraction": extraction_result,
                "q4_result": q4_result,
            }
        )

    leaderboard_entries = assign_leaderboard_ranks(
        [
            LeaderboardEntry(
                submission_id=submission["student_folder_name"],
                f1_score=submission["q4_result"].f1_score,
            )
            for submission in submission_payloads
            if submission["q4_result"].leaderboard_status == LeaderboardStatus.VALID
            and submission["q4_result"].f1_score is not None
        ]
    )
    ranked_lookup = {
        entry.submission_id: entry.rank
        for entry in leaderboard_entries
        if entry.rank is not None
    }

    summary_rows: list[dict[str, str]] = []
    for submission in submission_payloads:
        q4_result: Q4EvaluationResult = submission["q4_result"]
        rank = ranked_lookup.get(submission["student_folder_name"])
        if rank is not None:
            q4_result = q4_result.model_copy(update={"rank": rank})
            submission["q4_result"] = q4_result

        result_output_path = submissions_output_dir / f"{_slugify(submission['student_folder_name'])}.json"
        submission["result_json_path"] = result_output_path
        result_output_path.write_text(
            render_json_document(submission) + "\n",
            encoding="utf-8",
        )

        extraction_result: BatchExtractionResult = submission["extraction"]
        summary_rows.append(
            {
                "student_folder_name": submission["student_folder_name"],
                "student_folder_path": str(submission["student_folder_path"]),
                "zip_path": str(submission["zip_path"]) if submission["zip_path"] is not None else "",
                "extracted_submission_path": (
                    str(extraction_result.extracted_submission_path)
                    if extraction_result.extracted_submission_path is not None
                    else ""
                ),
                "execution_status": q4_result.execution_status.value,
                "leaderboard_status": q4_result.leaderboard_status.value,
                "predictions_valid": str(q4_result.predictions_valid).lower(),
                "requirements_env_used": str(q4_result.requirements_env_used).lower(),
                "f1_score": "" if q4_result.f1_score is None else str(q4_result.f1_score),
                "rank": "" if q4_result.rank is None else str(q4_result.rank),
                "zero_grade_policy_applied": str(q4_result.zero_grade_policy_applied).lower(),
                "zero_grade_policy_reason": q4_result.zero_grade_policy_reason or "",
                "failure_category": q4_result.failure_category.value if q4_result.failure_category else "",
                "failure_reason": q4_result.failure_reason or "",
                "result_json_path": str(result_output_path),
            }
        )

    summary_json_path = resolved_output_dir / "q4_summary.json"
    summary_csv_path = resolved_output_dir / "q4_summary.csv"
    leaderboard_csv_path = resolved_output_dir / "q4_leaderboard.csv"
    aggregate_payload = {
        **base_payload,
        "status": _overall_q4_batch_status(submission_payloads),
        "manifest": manifest,
        "submissions": submission_payloads,
        "summary_json_path": summary_json_path,
        "summary_csv_path": summary_csv_path,
        "leaderboard_csv_path": leaderboard_csv_path,
        "leaderboard": leaderboard_entries,
    }

    summary_json_path.write_text(
        render_json_document(aggregate_payload) + "\n",
        encoding="utf-8",
    )
    _write_q4_summary_csv(summary_csv_path, summary_rows)
    _write_q4_leaderboard_csv(
        leaderboard_csv_path,
        submission_payloads=submission_payloads,
        leaderboard_entries=leaderboard_entries,
    )

    exit_code = 1 if aggregate_payload["status"] == "failed" else 0
    return aggregate_payload, exit_code


def _overall_q23_status(results: dict[str, QuestionGradingResult]) -> str:
    if any(result.status == QuestionGradingStatus.FAILED for result in results.values()):
        return "failed"
    if any(result.status == QuestionGradingStatus.REVIEW for result in results.values()):
        return "review"
    return "scored"


def _merged_review_reasons(results: dict[str, QuestionGradingResult]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for result in results.values():
        for reason in result.review_reasons:
            if reason in seen:
                continue
            merged.append(reason)
            seen.add(reason)
    return merged


def _overall_batch_status(submissions: list[dict[str, Any]]) -> str:
    if any(str(submission["grading"].get("status")) == "failed" for submission in submissions):
        return "failed"
    if any(str(submission["grading"].get("status")) == "review" for submission in submissions):
        return "review"
    return "scored"


def _merge_batch_review_reasons(submissions: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for submission in submissions:
        for reason in submission["grading"].get("review_reasons", []):
            if not reason or reason in seen:
                continue
            merged.append(str(reason))
            seen.add(str(reason))
    return merged


def _write_batch_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "student_folder_name",
            "student_folder_path",
            "zip_path",
            "extracted_submission_path",
            "grading_status",
            "review_required",
            "review_reasons",
            "failure_reason",
            "result_json_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _prepare_batch_q4_submission(
    entry: BatchDiscoveryEntry,
    extract_root: Path,
) -> BatchExtractionResult:
    if entry.selected_zip_path is not None:
        return extract_submission_zip(entry, extract_root)
    warning_codes = {warning.code for warning in entry.warnings}
    if "multiple_zip_files_found" in warning_codes:
        return BatchExtractionResult(
            student_folder_name=entry.student_folder_name,
            student_folder_path=entry.student_folder_path,
            extraction_root=extract_root,
            status="failed",
            warnings=entry.warnings,
            failure_reason="Multiple zip files were found; Q4 batch execution requires a single unambiguous submission source.",
        )
    return BatchExtractionResult(
        student_folder_name=entry.student_folder_name,
        student_folder_path=entry.student_folder_path,
        extraction_root=extract_root,
        extracted_submission_path=entry.student_folder_path.resolve(),
        status="ready",
        warnings=entry.warnings,
    )


def _overall_q4_batch_status(submissions: list[dict[str, Any]]) -> str:
    if any(
        submission["q4_result"].execution_status != Q4ExecutionStatus.SUCCEEDED
        for submission in submissions
    ):
        return "failed"
    return "completed"


def _write_q4_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "student_folder_name",
            "student_folder_path",
            "zip_path",
            "extracted_submission_path",
            "execution_status",
            "leaderboard_status",
            "predictions_valid",
            "requirements_env_used",
            "f1_score",
            "rank",
            "zero_grade_policy_applied",
            "zero_grade_policy_reason",
            "failure_category",
            "failure_reason",
            "result_json_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_q4_leaderboard_csv(
    path: Path,
    *,
    submission_payloads: list[dict[str, Any]],
    leaderboard_entries: list[LeaderboardEntry],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_lookup = {
        submission["student_folder_name"]: submission
        for submission in submission_payloads
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rank",
            "submission_id",
            "student_folder_name",
            "student_folder_path",
            "extracted_submission_path",
            "f1_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in leaderboard_entries:
            submission = payload_lookup[entry.submission_id]
            extraction_result: BatchExtractionResult = submission["extraction"]
            writer.writerow(
                {
                    "rank": entry.rank,
                    "submission_id": entry.submission_id,
                    "student_folder_name": submission["student_folder_name"],
                    "student_folder_path": str(submission["student_folder_path"]),
                    "extracted_submission_path": (
                        str(extraction_result.extracted_submission_path)
                        if extraction_result.extracted_submission_path is not None
                        else ""
                    ),
                    "f1_score": entry.f1_score,
                }
            )


def _slugify(value: str) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in value.lower())
    compact_slug = "-".join(part for part in slug.split("-") if part)
    return compact_slug or "submission"
