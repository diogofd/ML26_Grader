from __future__ import annotations

import fnmatch
from pathlib import Path

from pydantic import BaseModel, ConfigDict

STRONG_AUXILIARY_NOTEBOOK_HINTS = (
    "saveaspickle",
    "pickle",
    "export",
    "deployment",
    "deploy",
    "modeltesting",
    "packaging",
    "package",
    "serializ",
    "serialis",
)
WEAK_AUXILIARY_NOTEBOOK_HINTS = (
    "validation",
    "testing",
    "helper",
    "artifact",
    "reload",
    "smoke",
)


class SubmissionArtifactPatterns(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notebook_glob: str = "*Complaints*Notebook*.ipynb"
    requirements_glob: str = "*_requirements.txt"
    feature_engineering_glob: str = "feature_engineering.py"
    combined_pipeline_glob: str = "*_Pipeline.pkl"
    split_preprocessor_glob: str = "*_Preprocessor.pkl"
    split_model_glob: str = "*_Model.pkl"


class DiscoveredSubmissionArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_root: Path
    notebooks: list[Path]
    requirements_files: list[Path]
    feature_engineering_files: list[Path]
    combined_pipeline_files: list[Path]
    split_preprocessor_files: list[Path]
    split_model_files: list[Path]


def is_discoverable_notebook(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".ipynb"
        and not _is_checkpoint_notebook(path)
    )


def notebook_analysis_preference_score(path: Path) -> int:
    normalized_name = _normalize_artifact_name(path.stem)
    score = 0
    if "complaint" in normalized_name:
        score += 3
    if "complaint" in normalized_name and "notebook" in normalized_name:
        score += 2
    if "analysis" in normalized_name:
        score += 2
    if "notebook" in normalized_name:
        score += 1
    if "assignment" in normalized_name:
        score += 1
    if "report" in normalized_name or "solution" in normalized_name:
        score += 1
    if any(hint in normalized_name for hint in STRONG_AUXILIARY_NOTEBOOK_HINTS):
        score -= 6
    if any(hint in normalized_name for hint in WEAK_AUXILIARY_NOTEBOOK_HINTS):
        score -= 4
    return score


def is_analysis_likely_notebook(path: Path) -> bool:
    return notebook_analysis_preference_score(path) >= 0


def _is_checkpoint_notebook(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    lowered_name = path.name.lower()
    return (
        ".ipynb_checkpoints" in lowered_parts
        or "__macosx" in lowered_parts
        or lowered_name.startswith("._")
        or lowered_name.endswith("-checkpoint.ipynb")
    )


def _pattern_matches(path: Path, root: Path, pattern: str) -> bool:
    lowered_pattern = pattern.lower()
    relative_path = path.relative_to(root).as_posix().lower()
    return fnmatch.fnmatch(relative_path, lowered_pattern) or fnmatch.fnmatch(
        path.name.lower(),
        lowered_pattern,
    )


def _normalize_artifact_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _sorted_matches(
    root: Path,
    pattern: str,
    *,
    notebook_only: bool = False,
) -> list[Path]:
    # Use case-insensitive glob matching so realistic student filenames are found
    # consistently across Windows and Linux runs.
    matches = (
        path
        for path in root.rglob("*")
        if path.is_file() and _pattern_matches(path, root, pattern)
    )
    if notebook_only:
        return sorted(path for path in matches if is_discoverable_notebook(path))
    return sorted(matches)


def scan_submission_directory(
    submission_root: Path,
    patterns: SubmissionArtifactPatterns | None = None,
) -> DiscoveredSubmissionArtifacts:
    root = submission_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Submission path does not exist: {submission_root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Submission path is not a directory: {submission_root}")

    active_patterns = patterns or SubmissionArtifactPatterns()
    return DiscoveredSubmissionArtifacts(
        submission_root=root,
        notebooks=_sorted_matches(
            root,
            active_patterns.notebook_glob,
            notebook_only=True,
        ),
        requirements_files=_sorted_matches(root, active_patterns.requirements_glob),
        feature_engineering_files=_sorted_matches(root, active_patterns.feature_engineering_glob),
        combined_pipeline_files=_sorted_matches(root, active_patterns.combined_pipeline_glob),
        split_preprocessor_files=_sorted_matches(root, active_patterns.split_preprocessor_glob),
        split_model_files=_sorted_matches(root, active_patterns.split_model_glob),
    )
