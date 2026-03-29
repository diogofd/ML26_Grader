from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .models import FailureCategory, Q4ArtifactLayout, Q4ExecutionStatus


class Q4ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_root: Path
    artifact_layout: Q4ArtifactLayout
    dataset_name: str = Field(min_length=1)
    dataset_path: Path
    timeout_seconds: int = Field(ge=1)
    input_row_count: int = Field(ge=1)


class Q4ExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(default="disabled", min_length=1)
    execution_status: Q4ExecutionStatus
    predictions: list[int | bool | str | float] = Field(default_factory=list)
    requirements_env_used: bool = False
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None
    execution_logs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.execution_status == Q4ExecutionStatus.FAILED:
            if self.failure_category is None or not self.failure_reason:
                raise ValueError("Failed execution responses require failure details.")
        if self.execution_status == Q4ExecutionStatus.SUCCEEDED and self.failure_category is not None:
            raise ValueError("Successful execution responses cannot include failure details.")
        if self.failure_category is None and self.failure_reason is not None:
            raise ValueError("failure_reason requires failure_category.")
        if self.failure_category is not None and not self.failure_reason:
            raise ValueError("failure_category requires failure_reason.")
        return self


class Q4ExecutionBackend(Protocol):
    def run(self, request: Q4ExecutionRequest) -> Q4ExecutionResponse:
        ...


class _RequirementsEnvFailure(Exception):
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


class DisabledQ4ExecutionBackend:
    # Actual student-code execution belongs behind a dedicated subprocess backend.
    def run(self, request: Q4ExecutionRequest) -> Q4ExecutionResponse:
        return Q4ExecutionResponse(
            backend_name="disabled",
            execution_status=Q4ExecutionStatus.NOT_RUN,
            failure_category=FailureCategory.EXECUTION_DISABLED,
            failure_reason=(
                "Student-code execution is disabled because no Q4 execution backend is configured."
            ),
            execution_logs=[
                f"Prepared Q4 execution request for {request.dataset_name} with {request.input_row_count} rows.",
                "No subprocess backend was configured, so student artifacts were not loaded or executed.",
            ],
        )


class SubprocessQ4ExecutionBackend:
    def __init__(
        self,
        *,
        python_executable: str | None = None,
        worker_module: str = "ml26_grader.q4.worker",
        package_src_root: Path | None = None,
        use_submission_requirements: bool = False,
        requirements_env_root: Path | None = None,
        requirements_install_timeout_seconds: int = 600,
        requirements_reuse_envs: bool = True,
    ) -> None:
        self._python_executable = python_executable or sys.executable
        self._worker_module = worker_module
        self._package_src_root = (
            package_src_root.resolve()
            if package_src_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self._use_submission_requirements = use_submission_requirements
        self._requirements_env_root = (
            requirements_env_root.resolve()
            if requirements_env_root is not None
            else self._package_src_root.parent / "sandbox" / "q4_requirements_envs"
        )
        self._requirements_install_timeout_seconds = requirements_install_timeout_seconds
        self._requirements_reuse_envs = requirements_reuse_envs

    def run(self, request: Q4ExecutionRequest) -> Q4ExecutionResponse:
        requirements_logs: list[str] = []
        requirements_env_used = False
        python_executable = self._python_executable

        if self._use_submission_requirements:
            try:
                python_executable, requirements_logs = self._resolve_requirements_python(request)
                requirements_env_used = True
            except _RequirementsEnvFailure as exc:
                return Q4ExecutionResponse(
                    backend_name="subprocess",
                    execution_status=Q4ExecutionStatus.FAILED,
                    requirements_env_used=False,
                    failure_category=exc.category,
                    failure_reason=exc.reason,
                    execution_logs=exc.logs,
                )

        with tempfile.TemporaryDirectory(prefix="ml26-grader-q4-") as temp_dir:
            temp_root = Path(temp_dir)
            request_path = temp_root / "request.json"
            response_path = temp_root / "response.json"
            request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

            command = [
                python_executable,
                "-m",
                self._worker_module,
                str(request_path),
                str(response_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=request.submission_root,
                    env=self._build_worker_env(request.submission_root),
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return Q4ExecutionResponse(
                    backend_name="subprocess",
                    execution_status=Q4ExecutionStatus.FAILED,
                    requirements_env_used=requirements_env_used,
                    failure_category=FailureCategory.TIMEOUT,
                    failure_reason=(
                        f"Q4 subprocess exceeded the timeout of {request.timeout_seconds} seconds."
                    ),
                    execution_logs=[
                        *requirements_logs,
                        *self._subprocess_logs(command, exc.stdout, exc.stderr),
                    ],
                )

            if completed.returncode != 0:
                failure_category = self._bootstrap_failure_category(
                    completed.stdout,
                    completed.stderr,
                )
                return Q4ExecutionResponse(
                    backend_name="subprocess",
                    execution_status=Q4ExecutionStatus.FAILED,
                    requirements_env_used=requirements_env_used,
                    failure_category=failure_category,
                    failure_reason=(
                        "Q4 subprocess exited unexpectedly "
                        f"with code {completed.returncode}."
                    ),
                    execution_logs=[
                        *requirements_logs,
                        *self._subprocess_logs(
                            command,
                            completed.stdout,
                            completed.stderr,
                        ),
                    ],
                )

            try:
                response = Q4ExecutionResponse.model_validate_json(
                    response_path.read_text(encoding="utf-8")
                )
            except FileNotFoundError:
                return Q4ExecutionResponse(
                    backend_name="subprocess",
                    execution_status=Q4ExecutionStatus.FAILED,
                    requirements_env_used=requirements_env_used,
                    failure_category=self._bootstrap_failure_category(
                        completed.stdout,
                        completed.stderr,
                    ),
                    failure_reason="Q4 subprocess did not produce a response file.",
                    execution_logs=[
                        *requirements_logs,
                        *self._subprocess_logs(
                            command,
                            completed.stdout,
                            completed.stderr,
                        ),
                    ],
                )
            except (ValidationError, ValueError) as exc:
                return Q4ExecutionResponse(
                    backend_name="subprocess",
                    execution_status=Q4ExecutionStatus.FAILED,
                    requirements_env_used=requirements_env_used,
                    failure_category=self._bootstrap_failure_category(
                        completed.stdout,
                        completed.stderr,
                    ),
                    failure_reason=f"Q4 subprocess returned invalid structured output: {exc}",
                    execution_logs=[
                        *requirements_logs,
                        *self._subprocess_logs(
                            command,
                            completed.stdout,
                            completed.stderr,
                        ),
                    ],
                )

            merged_logs = [
                *requirements_logs,
                f"Executed Q4 subprocess backend from {request.submission_root}.",
                *self._subprocess_logs(command, completed.stdout, completed.stderr),
                *response.execution_logs,
            ]
            return response.model_copy(
                update={
                    "requirements_env_used": requirements_env_used,
                    "execution_logs": merged_logs,
                }
            )

    def _resolve_requirements_python(self, request: Q4ExecutionRequest) -> tuple[str, list[str]]:
        requirements_file = request.artifact_layout.requirements_file
        if requirements_file is None:
            raise _RequirementsEnvFailure(
                FailureCategory.MISSING_ARTIFACTS,
                "requirements-aware Q4 execution was enabled but requirements.txt was missing.",
                logs=["requirements-aware Q4 execution could not start because requirements.txt was missing."],
            )

        env_key = self._requirements_env_key(requirements_file)
        env_root = self._requirements_env_root / env_key
        env_logs = [
            f"requirements-aware Q4 execution enabled for {requirements_file}.",
            f"Requirements environment key: {env_key}.",
            f"Requirements environment root: {env_root}.",
        ]

        if not self._requirements_reuse_envs and env_root.exists():
            shutil.rmtree(env_root, ignore_errors=True)
            env_logs.append("Removed existing requirements environment because reuse is disabled.")

        if self._requirements_reuse_envs and self._is_requirements_env_ready(env_root):
            env_logs.append("Reusing existing requirements environment for this submission.")
            return str(self._venv_python_executable(env_root)), env_logs

        if env_root.exists():
            shutil.rmtree(env_root, ignore_errors=True)
            env_logs.append("Removed incomplete requirements environment before recreation.")

        env_root.parent.mkdir(parents=True, exist_ok=True)
        self._create_virtual_environment(env_root, env_logs)
        env_python = self._venv_python_executable(env_root)
        self._install_submission_requirements(env_python, requirements_file, env_logs)
        self._requirements_ready_marker(env_root).write_text(env_key + "\n", encoding="utf-8")
        env_logs.append("Prepared submission-specific requirements environment successfully.")
        return str(env_python), env_logs

    def _requirements_env_key(self, requirements_file: Path) -> str:
        digest = hashlib.sha256(requirements_file.read_bytes()).hexdigest()[:16]
        return f"py{sys.version_info.major}{sys.version_info.minor}-{digest}"

    def _create_virtual_environment(self, env_root: Path, logs: list[str]) -> None:
        command = [
            self._python_executable,
            "-m",
            "venv",
            str(env_root),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._requirements_install_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise _RequirementsEnvFailure(
                FailureCategory.REQUIREMENTS_ENV_CREATION_FAILED,
                (
                    "Creating the submission-specific requirements environment exceeded "
                    f"{self._requirements_install_timeout_seconds} seconds."
                ),
                logs=[*logs, *self._subprocess_logs(command, exc.stdout, exc.stderr)],
            ) from exc

        if completed.returncode != 0:
            raise _RequirementsEnvFailure(
                FailureCategory.REQUIREMENTS_ENV_CREATION_FAILED,
                (
                    "Creating the submission-specific requirements environment failed "
                    f"with exit code {completed.returncode}."
                ),
                logs=[*logs, *self._subprocess_logs(command, completed.stdout, completed.stderr)],
            )
        logs.extend(self._subprocess_logs(command, completed.stdout, completed.stderr))

    def _install_submission_requirements(
        self,
        env_python: Path,
        requirements_file: Path,
        logs: list[str],
    ) -> None:
        command = [
            str(env_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "-r",
            str(requirements_file),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._requirements_install_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise _RequirementsEnvFailure(
                FailureCategory.REQUIREMENTS_INSTALL_FAILED,
                (
                    f"Installing requirements from {requirements_file} exceeded "
                    f"{self._requirements_install_timeout_seconds} seconds."
                ),
                logs=[*logs, *self._subprocess_logs(command, exc.stdout, exc.stderr)],
            ) from exc

        if completed.returncode != 0:
            raise _RequirementsEnvFailure(
                FailureCategory.REQUIREMENTS_INSTALL_FAILED,
                (
                    f"Installing requirements from {requirements_file} failed "
                    f"with exit code {completed.returncode}."
                ),
                logs=[*logs, *self._subprocess_logs(command, completed.stdout, completed.stderr)],
            )
        logs.extend(self._subprocess_logs(command, completed.stdout, completed.stderr))

    def _is_requirements_env_ready(self, env_root: Path) -> bool:
        return self._requirements_ready_marker(env_root).exists() and self._venv_python_executable(
            env_root
        ).exists()

    def _requirements_ready_marker(self, env_root: Path) -> Path:
        return env_root / ".ml26_grader_requirements_ready"

    def _venv_python_executable(self, env_root: Path) -> Path:
        if os.name == "nt":
            return env_root / "Scripts" / "python.exe"
        return env_root / "bin" / "python"

    def _build_worker_env(self, submission_root: Path) -> dict[str, str]:
        env = dict(os.environ)
        pythonpath_entries = self._pythonpath_entries(submission_root, env.get("PYTHONPATH"))
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        return env

    def _pythonpath_entries(
        self,
        submission_root: Path,
        existing_pythonpath: str | None,
    ) -> list[str]:
        raw_entries: list[str] = [
            str(submission_root),
            str(self._package_src_root),
        ]
        raw_entries.extend(
            str(Path(entry).resolve()) if entry else str(Path.cwd().resolve())
            for entry in sys.path
            if entry is not None
        )
        if existing_pythonpath:
            raw_entries.extend(
                entry
                for entry in existing_pythonpath.split(os.pathsep)
                if entry
            )

        deduped_entries: list[str] = []
        seen: set[str] = set()
        for entry in raw_entries:
            normalized = str(Path(entry).resolve()) if entry else str(Path.cwd().resolve())
            if normalized in seen:
                continue
            deduped_entries.append(normalized)
            seen.add(normalized)
        return deduped_entries

    def _bootstrap_failure_category(
        self,
        stdout: str | None,
        stderr: str | None,
    ) -> FailureCategory:
        combined_output = "\n".join(part for part in (stdout, stderr) if part).lower()
        if "modulenotfounderror" in combined_output or "importerror" in combined_output:
            return FailureCategory.IMPORT_FAILURE
        return FailureCategory.INFERENCE_FAILURE

    def _subprocess_logs(
        self,
        command: list[str],
        stdout: str | None,
        stderr: str | None,
    ) -> list[str]:
        logs = [
            "Launched Q4 subprocess command: " + " ".join(command),
        ]
        if stdout:
            logs.extend(
                f"subprocess_stdout: {line}"
                for line in stdout.splitlines()
                if line.strip()
            )
        if stderr:
            logs.extend(
                f"subprocess_stderr: {line}"
                for line in stderr.splitlines()
                if line.strip()
            )
        return logs
