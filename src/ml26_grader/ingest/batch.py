from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BatchWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class BatchDiscoveryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_folder_name: str = Field(min_length=1)
    student_folder_path: Path
    zip_files: list[Path] = Field(default_factory=list)
    selected_zip_path: Path | None = None
    warnings: list[BatchWarning] = Field(default_factory=list)

    @property
    def ready_for_extraction(self) -> bool:
        return self.selected_zip_path is not None and not any(
            warning.code in {"no_zip_found", "multiple_zip_files_found"}
            for warning in self.warnings
        )


class BatchDiscoveryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_root: Path
    submissions: list[BatchDiscoveryEntry] = Field(default_factory=list)
    warnings: list[BatchWarning] = Field(default_factory=list)


class BatchExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_folder_name: str = Field(min_length=1)
    student_folder_path: Path
    zip_path: Path | None = None
    extraction_root: Path
    extracted_archive_root: Path | None = None
    extracted_submission_path: Path | None = None
    status: Literal["ready", "failed"]
    warnings: list[BatchWarning] = Field(default_factory=list)
    failure_reason: str | None = None


def discover_batch_submissions(batch_root: Path) -> BatchDiscoveryManifest:
    resolved_batch_root = batch_root.resolve()
    if not resolved_batch_root.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_root}")
    if not resolved_batch_root.is_dir():
        raise NotADirectoryError(f"Batch directory is not a directory: {batch_root}")

    entries: list[BatchDiscoveryEntry] = []
    manifest_warnings: list[BatchWarning] = []

    for child in sorted(resolved_batch_root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            manifest_warnings.append(
                BatchWarning(
                    code="top_level_non_directory_ignored",
                    message=f"Ignored non-directory entry at batch root: {child.name}.",
                )
            )
            continue

        zip_files = sorted(
            candidate
            for candidate in child.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == ".zip"
        )
        non_zip_files = sorted(
            candidate
            for candidate in child.iterdir()
            if candidate.is_file() and candidate.suffix.lower() != ".zip"
        )

        warnings: list[BatchWarning] = []
        selected_zip_path: Path | None = None

        if not zip_files:
            warnings.append(
                BatchWarning(
                    code="no_zip_found",
                    message=f"No zip file was found in {child.name}.",
                )
            )
        elif len(zip_files) > 1:
            warnings.append(
                BatchWarning(
                    code="multiple_zip_files_found",
                    message=(
                        f"Multiple zip files were found in {child.name}; manual review is required."
                    ),
                )
            )
        else:
            selected_zip_path = zip_files[0]

        if non_zip_files:
            warnings.append(
                BatchWarning(
                    code="non_zip_files_present",
                    message=f"Non-zip files were present alongside submission artifacts in {child.name}.",
                )
            )

        entries.append(
            BatchDiscoveryEntry(
                student_folder_name=child.name,
                student_folder_path=child,
                zip_files=zip_files,
                selected_zip_path=selected_zip_path,
                warnings=warnings,
            )
        )

    return BatchDiscoveryManifest(
        batch_root=resolved_batch_root,
        submissions=entries,
        warnings=manifest_warnings,
    )


def extract_submission_zip(
    entry: BatchDiscoveryEntry,
    extraction_root: Path,
) -> BatchExtractionResult:
    resolved_extraction_root = extraction_root.resolve()
    resolved_extraction_root.mkdir(parents=True, exist_ok=True)

    if entry.selected_zip_path is None:
        return BatchExtractionResult(
            student_folder_name=entry.student_folder_name,
            student_folder_path=entry.student_folder_path,
            extraction_root=resolved_extraction_root,
            status="failed",
            warnings=entry.warnings,
            failure_reason="No unique submission zip was available for extraction.",
        )

    archive_root = resolved_extraction_root / _slugify(entry.student_folder_name)
    if archive_root.exists():
        _clear_directory(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(entry.selected_zip_path) as archive:
            _extract_archive_members(archive, archive_root)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        return BatchExtractionResult(
            student_folder_name=entry.student_folder_name,
            student_folder_path=entry.student_folder_path,
            zip_path=entry.selected_zip_path,
            extraction_root=resolved_extraction_root,
            extracted_archive_root=archive_root,
            status="failed",
            warnings=entry.warnings,
            failure_reason=f"Zip extraction failed: {exc}",
        )

    extracted_submission_path = _unwrap_submission_root(archive_root)
    return BatchExtractionResult(
        student_folder_name=entry.student_folder_name,
        student_folder_path=entry.student_folder_path,
        zip_path=entry.selected_zip_path,
        extraction_root=resolved_extraction_root,
        extracted_archive_root=archive_root,
        extracted_submission_path=extracted_submission_path,
        status="ready",
        warnings=entry.warnings,
    )


def _extract_archive_members(archive: zipfile.ZipFile, destination_root: Path) -> None:
    destination_root_resolved = destination_root.resolve()
    for info in archive.infolist():
        relative_path = _validated_archive_path(info.filename)
        target_path = destination_root.joinpath(*relative_path.parts)
        target_path_resolved = target_path.resolve()
        if destination_root_resolved not in (target_path_resolved, *target_path_resolved.parents):
            raise ValueError(f"Archive member escapes extraction root: {info.filename}")

        if info.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source_handle, target_path.open("wb") as destination_handle:
            destination_handle.write(source_handle.read())


def _validated_archive_path(name: str) -> PurePosixPath:
    relative_path = PurePosixPath(name.replace("\\", "/"))
    if not name or relative_path.is_absolute():
        raise ValueError(f"Invalid archive member path: {name!r}")
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return relative_path


def _unwrap_submission_root(archive_root: Path) -> Path:
    current_root = archive_root
    while True:
        meaningful_children = [
            child
            for child in current_root.iterdir()
            if child.name not in {"__MACOSX", ".DS_Store"}
        ]
        if len(meaningful_children) != 1 or not meaningful_children[0].is_dir():
            return current_root
        current_root = meaningful_children[0]


def _clear_directory(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _clear_directory(child)
            child.rmdir()
            continue
        child.unlink()


def _slugify(value: str) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in value.lower())
    compact_slug = "-".join(part for part in slug.split("-") if part)
    return compact_slug or "submission"
