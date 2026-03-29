from __future__ import annotations

import zipfile
from pathlib import Path

import nbformat

from ml26_grader.ingest.batch import discover_batch_submissions, extract_submission_zip


def _write_notebook(path: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "## Q2.1\nThis is a supervised binary classification problem."
            ),
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def _write_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for member_name, member_content in members.items():
            archive.writestr(member_name, member_content)
    return path


def test_discovers_single_zip_student_folder(tmp_path: Path) -> None:
    student_dir = tmp_path / "Student One"
    student_dir.mkdir()
    zip_path = _write_zip(student_dir / "submission.zip", {"wrapper/readme.txt": "hello"})

    manifest = discover_batch_submissions(tmp_path)

    assert manifest.batch_root == tmp_path.resolve()
    assert len(manifest.submissions) == 1
    entry = manifest.submissions[0]
    assert entry.student_folder_name == "Student One"
    assert entry.selected_zip_path == zip_path.resolve()
    assert entry.ready_for_extraction is True
    assert entry.warnings == []


def test_detects_zero_zip_and_multiple_zip_cases(tmp_path: Path) -> None:
    no_zip_dir = tmp_path / "No Zip Student"
    no_zip_dir.mkdir()
    (no_zip_dir / "notes.txt").write_text("not a zip", encoding="utf-8")

    multi_zip_dir = tmp_path / "Multi Zip Student"
    multi_zip_dir.mkdir()
    _write_zip(multi_zip_dir / "a.zip", {"a.txt": "A"})
    _write_zip(multi_zip_dir / "b.zip", {"b.txt": "B"})

    manifest = discover_batch_submissions(tmp_path)
    entries = {entry.student_folder_name: entry for entry in manifest.submissions}

    no_zip_entry = entries["No Zip Student"]
    assert no_zip_entry.selected_zip_path is None
    assert {warning.code for warning in no_zip_entry.warnings} == {
        "no_zip_found",
        "non_zip_files_present",
    }

    multi_zip_entry = entries["Multi Zip Student"]
    assert multi_zip_entry.selected_zip_path is None
    assert {warning.code for warning in multi_zip_entry.warnings} == {
        "multiple_zip_files_found",
    }


def test_safe_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    student_dir = tmp_path / "Traversal Student"
    student_dir.mkdir()
    zip_path = _write_zip(student_dir / "submission.zip", {"../escaped.txt": "nope"})
    entry = discover_batch_submissions(tmp_path).submissions[0]
    extraction_root = tmp_path / "extracted"

    result = extract_submission_zip(entry, extraction_root)

    assert zip_path.exists()
    assert result.status == "failed"
    assert "Zip extraction failed" in (result.failure_reason or "")
    assert not (extraction_root.parent / "escaped.txt").exists()


def test_wrapper_folder_zip_contents_are_unwrapped_to_submission_path(tmp_path: Path) -> None:
    student_dir = tmp_path / "Wrapper Student"
    student_dir.mkdir()
    notebook_path = _write_notebook(tmp_path / "StudentComplaintsNotebook.ipynb")
    with zipfile.ZipFile(student_dir / "submission.zip", "w") as archive:
        archive.write(
            notebook_path,
            arcname="assignsubmission_file/StudentComplaintsNotebook.ipynb",
        )

    entry = discover_batch_submissions(tmp_path).submissions[0]
    result = extract_submission_zip(entry, tmp_path / "extracted")

    assert result.status == "ready"
    assert result.extracted_archive_root is not None
    assert result.extracted_submission_path is not None
    assert result.extracted_submission_path.name == "assignsubmission_file"
    assert (result.extracted_submission_path / "StudentComplaintsNotebook.ipynb").exists()
