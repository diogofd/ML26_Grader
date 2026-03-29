# RUNBOOK.md

## Purpose

This runbook explains how to set up, test, and operate the ML26 grading pipeline.

Current scope:

- **Q2/Q3**: notebook evidence extraction + LLM grading
- **Q4**: deterministic subprocess execution + F1 + leaderboard-ready outputs
- **Batch mode**: Moodle-style student folders containing zip submissions, plus already extracted student folders for Q4 batch runs

For grading policy and judge behavior, see:

- `GRADING_POLICY.md`
- `LLM_JUDGE_SPEC.md`

---

## Pre-flight checklist

Before any real grading run, confirm all of the following:

- the WSL virtual environment is active
- `python -m pytest -q` passes
- `OPENAI_API_KEY` is set for the current shell
- `config/grading.toml` exists and points to the correct runtime settings
- `specs/questions.toml` and `specs/rubrics.toml` exist
- required runtime dependencies such as `pandas`, `pytest`, and `nbformat` are installed in the active venv
- the intended output directory is fresh or uniquely named for this run
- for Q4 runs, the configured dataset path exists and is the intended one
- for Q4 runs, non-functional submissions receive Q4 = 0 by policy; this includes missing required artifacts, load/import failures, inference failures, empty predictions, count mismatches, and invalid prediction formats
- for batch runs, the input folder is the intended cohort/sample and not the wrong directory

### Quick pre-flight commands

```bash
source .venv/bin/activate
python -m pytest -q
python - <<'PY'
import os
print("OPENAI_API_KEY:", "set" if os.getenv("OPENAI_API_KEY") else "missing")
PY
ls config/grading.toml specs/questions.toml specs/rubrics.toml
```

## 1. Environment setup

This project should be run from **WSL/Linux**, not from a Windows-created virtual environment.

### 1.1 Create the venv inside WSL

From the repo root:

```bash
deactivate 2>/dev/null || true
rm -rf .venv

sudo apt update
sudo apt install -y python3-venv python3-full python-is-python3

python3 -m venv .venv
source .venv/bin/activate
hash -r
```

Check that the venv is correct:

```bash
which python
which pip
```

Expected output should point to:

- `.venv/bin/python`
- `.venv/bin/pip`

### 1.2 Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -U pytest pandas nbformat
```

If student Q4 artifacts require extra ML libraries, install them into this environment as needed.

---

## 2. OpenAI API setup

Live Q2/Q3 grading requires an API key.

Set it in the current shell:

```bash
export OPENAI_API_KEY='sk-...'
```

Verify it:

```bash
python - <<'PY'
import os
k = os.getenv("OPENAI_API_KEY")
print("missing" if not k else ("placeholder" if "your_key" in k else f"set len={len(k)}"))
PY
```

You want:

```text
set len=...
```

Do **not** commit API keys to the repo.

---

## 3. Runtime files

Live runs expect:

- `config/grading.toml`
- `specs/questions.toml`
- `specs/rubrics.toml`

Templates/examples:

- `config/grading.example.toml`
- `specs/rubrics.example.toml`

Before running, confirm:

- `llm.enabled = true` in `config/grading.toml` for Q2/Q3 live grading
- model/provider settings are correct
- Q4 dataset paths are correct

---

## 4. Test commands

### Full suite

```bash
python -m pytest -q
```

### Q2/Q3-focused tests

```bash
python -m pytest -q \
  tests/test_extraction_service.py \
  tests/test_q23_grading_pipeline.py \
  tests/test_openai_judge_adapter.py
```

### Q4-focused tests

```bash
python -m pytest -q \
  tests/test_q4_pipeline.py \
  tests/test_cli_q4.py \
  tests/test_cli_batch_q4.py
```

### Batch intake tests

```bash
python -m pytest -q \
  tests/test_ingest_batch.py \
  tests/test_cli_batch_q23.py
```

---

## 5. Single-submission commands

### Validate config

```bash
python -m ml26_grader validate-config config/grading.toml
```

### Extract notebook evidence only

```bash
python -m ml26_grader extract-evidence /path/to/extracted_submission Q2
python -m ml26_grader extract-evidence /path/to/extracted_submission Q3
```

### Grade Q2/Q3 for one extracted submission

```bash
python -m ml26_grader grade-q23-submission /path/to/extracted_submission \
  --config ./config/grading.toml \
  --questions ./specs/questions.toml \
  --rubrics ./specs/rubrics.toml \
  --output ./out/single_q23.json
```

### Inspect Q4 for one extracted submission

```bash
python -m ml26_grader inspect-q4 /path/to/extracted_submission \
  --config ./config/grading.toml \
  --dataset modeltesting
```

---

## 6. Batch commands

### Batch Q2/Q3

For Moodle-style folders:

```bash
python -m ml26_grader grade-q23-batch ./ASSIGNMENTSML26 \
  --config ./config/grading.toml \
  --questions ./specs/questions.toml \
  --rubrics ./specs/rubrics.toml \
  --extract-root ./sandbox/extracted_q23 \
  --output-dir ./out/q23_batch_run_01
```

Outputs:

- `batch_summary.json`
- `batch_summary.csv`
- per-submission JSON under `output_dir/submissions/`

### Batch Q4

This batch path supports Moodle-style zipped folders and already extracted student folders.

```bash
python -m ml26_grader grade-q4-batch ./ASSIGNMENTSML26 \
  --config ./config/grading.toml \
  --extract-root ./sandbox/extracted_q4 \
  --output-dir ./out/q4_batch_run_01 \
  --dataset modeltesting
```

Outputs:

- `q4_summary.json`
- `q4_summary.csv`
- `q4_leaderboard.csv`
- per-submission JSON under `output_dir/submissions/`

---

## 7. Expected batch input shape

Typical Moodle-style batch input:

```text
ASSIGNMENTSML26/
  Student Name_123_assignsubmission_file/
    71078_Complaints_Assignment.zip
  Another Student_124_assignsubmission_file/
    71079_Complaints_Assignment.zip
```

The batch layer:

- discovers student folders
- detects zip files
- safely extracts valid zips
- reuses the single-submission grading/evaluation pipelines

Q4 batch can also evaluate already extracted student folders directly when no zip is present.

---

## 8. Output files to inspect

### Q2/Q3 batch

Inspect:

- `batch_summary.csv`
- `batch_summary.json`
- `submissions/*.json`

Useful fields:

- `status`
- `review_required`
- `review_tier`
- `hard_review_reasons`
- `soft_review_reasons`
- `soft_auto_pass_applied`
- `score_summary`

### Q4 batch

Inspect:

- `q4_summary.csv`
- `q4_summary.json`
- `q4_leaderboard.csv`
- `submissions/*.json`

Useful fields:

- `execution_status`
- `zero_grade_policy_applied`
- `zero_grade_policy_reason`
- `failure_category`
- `failure_reason`
- `predictions_valid`
- `f1_score`
- `rank`

---

## 9. Review workflow

### Q2/Q3

Current intent:

- strong, coherent submissions may auto-pass
- borderline or warning-heavy submissions go to review
- broken or unusable submissions fail closed

Review the per-submission JSON for:

- extracted evidence
- extraction warnings
- student-facing feedback
- internal notes
- judge audit metadata

### Q4

Review failures for:

- missing artifacts
- non-functional submission outcomes causing an explicit zero-by-policy result
- dependency/import failures
- prediction count mismatch
- invalid binary outputs
- model execution failures

Only valid runs participate in the leaderboard.

---

## 10. Known limitations

### Q2/Q3

- Notebook evidence extraction is heuristic.
- Helper notebooks can still occasionally compete with analysis notebooks.
- Review rate may remain above the long-term target on messy submissions.
- Strong markdown-only answers may still route to review.

### Q4

- Execution uses subprocess isolation, **not** true sandboxing.
- Student pickles and `feature_engineering.py` remain unsafe at the OS/security level.
- The runtime environment must already contain required dependencies.
- `requirements.txt` is not automatically provisioned per submission.
- Non-functional Q4 submissions are treated as policy-zero outcomes, not recoverable warnings.

### General

- The system is suitable for supervised pilot use.
- It is not yet a hardened multi-tenant sandbox.

---

## 11. Troubleshooting

### Wrong pytest version is being used

Example symptom:

- repo requires pytest 8.x
- system pytest 7.x is used instead

Fix:

```bash
python -m pip install -U pytest
python -m pytest -q
```

Prefer `python -m pytest`, not plain `pytest`.

### Active `.venv` but `python` points to `/usr/bin/python`

Cause:

- `.venv` was created on Windows, not in WSL

Fix:

- delete `.venv`
- recreate it inside WSL
- reinstall dependencies

### OpenAI 401 / invalid key

Check:

```bash
python - <<'PY'
import os
k = os.getenv("OPENAI_API_KEY")
print("missing" if not k else ("placeholder" if "your_key" in k else f"set len={len(k)}"))
PY
```

If it prints `placeholder`, the shell still has a dummy key.

### Q4 returns `import_failure: No module named 'pandas'`

Install pandas in the active venv:

```bash
python -m pip install -U pandas
```

### Batch run ends overall as `failed`

Inspect:

- summary JSON/CSV
- per-submission JSON
- whether failures are due to:
  - extraction failures
  - provider failures
  - Q4 dependency/import failures
  - invalid predictions

The batch can end as `"failed"` even when many submissions were processed successfully.

---

## 12. Recommended operational workflow

### Small pilot

1. Run the full test suite
2. Run Q2/Q3 batch on 10–15 submissions
3. Inspect review rate and representative JSONs
4. Run Q4 batch on the same set
5. Inspect leaderboard and failure modes

### Full grading run

1. Freeze the code
2. Run the full test suite
3. Run Q2/Q3 batch
4. Review flagged cases
5. Run Q4 batch
6. Review Q4 failures
7. Export summaries and leaderboard
8. Produce final combined grading outputs externally if needed

---

## 13. Suggested local folder conventions

Use scratch/output folders that are easy to clean and do not get committed:

```text
sandbox/
  extracted_q23/
  extracted_q4/
out/
  q23_batch_run_01/
  q4_batch_run_01/
```

Add local scratch/output folders to `.gitignore` if needed.

---

## 14. Current status summary

At the current state of the project:

- Q2/Q3 single and batch grading work
- OpenAI-backed grading works
- soft auto-pass for strong cases exists
- Q4 single and batch evaluation work
- leaderboard generation exists
- the system is ready for supervised pilot use
- remaining work is mostly calibration, operator ergonomics, and hardening
