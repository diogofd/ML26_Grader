# ML26 Grader

Reusable grading pipeline for Assignment 1, focused on Q2, Q3, and Q4.

This repository is designed for supervised grading workflows with a strong audit trail:

- `Q2` and `Q3` are graded from extracted evidence packets and a rubric-based LLM judge.
- `Q4` is graded deterministically from submission artifacts and evaluation outputs.
- Batch mode supports Moodle-style submission folders.

## Scope

In scope:

- `Q2`
- `Q3`
- `Q4`
- audit-friendly grading outputs
- batch processing for submission folders

Out of scope:

- `Q1`
- `Q5`
- plagiarism detection
- late penalties
- misconduct workflows
- final publication UI

## Source Of Truth

Read these first:

1. `GRADING_POLICY.md`
2. `LLM_JUDGE_SPEC.md`
3. `RUNBOOK.md`

If code and docs disagree, the policy/spec docs win.

## Core Design Rules

- `Q4` must remain fully deterministic.
- `Q4` must not use an LLM.
- `Q2` and `Q3` must be judged from extracted evidence packets, not freeform whole-notebook reading.
- Scoring rules and point weights come from config/specs, not hard-coded assumptions.
- The grader should fail closed instead of guessing.

## Repository Layout

```text
config/      Runtime config templates and local grader config
data/        Public datasets used by the grader
reference/   Assignment materials and reusable notebook fixtures
specs/       Question text and rubric definitions
src/         Python package implementation
tests/       Automated tests
RUNBOOK.md   Detailed setup and operational guidance
```

## Prerequisites

- Python `3.11` or `3.12`
- Git LFS
- A virtual environment created in WSL/Linux for normal operation
- `OPENAI_API_KEY` for live Q2/Q3 grading

This repo tracks the large public datasets with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

## Quick Start

The detailed setup flow lives in `RUNBOOK.md`. The short version is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Create a runtime config from the example:

```bash
cp config/grading.example.toml config/grading.toml
```

Then update:

- official point values
- provider/model settings
- any local runtime paths

For live Q2/Q3 grading, make sure `llm.enabled = true` and `OPENAI_API_KEY` is set.

## Main Commands

Validate config:

```bash
python -m ml26_grader validate-config config/grading.toml
```

List bundled public datasets:

```bash
python -m ml26_grader list-datasets
```

Extract notebook evidence for one question:

```bash
python -m ml26_grader extract-evidence /path/to/extracted_submission Q2
python -m ml26_grader extract-evidence /path/to/extracted_submission Q3
```

Grade one extracted submission for Q2/Q3:

```bash
python -m ml26_grader grade-q23-submission /path/to/extracted_submission \
  --config ./config/grading.toml \
  --questions ./specs/questions.toml \
  --rubrics ./specs/rubrics.toml
```

Inspect one extracted submission for Q4:

```bash
python -m ml26_grader inspect-q4 /path/to/extracted_submission \
  --config ./config/grading.toml \
  --dataset modeltesting
```

Batch grade Q2/Q3:

```bash
python -m ml26_grader grade-q23-batch ./ASSIGNMENTSML26 \
  --config ./config/grading.toml \
  --questions ./specs/questions.toml \
  --rubrics ./specs/rubrics.toml \
  --extract-root ./sandbox/extracted_q23 \
  --output-dir ./out/q23_batch_run_01
```

Batch evaluate Q4:

```bash
python -m ml26_grader grade-q4-batch ./ASSIGNMENTSML26 \
  --config ./config/grading.toml \
  --extract-root ./sandbox/extracted_q4 \
  --output-dir ./out/q4_batch_run_01 \
  --dataset modeltesting
```

## Expected Outputs

For `Q2/Q3`, the grader produces question/subquestion scores, confidence, feedback, notes, and review signals.

For `Q4`, the grader produces deterministic execution results including:

- `f1_score`
- `rank`
- `leaderboard_status`
- execution status
- failure reasons when applicable

Batch runs produce summary files plus per-submission JSON outputs.

## Testing

Run the full suite:

```bash
python -m pytest -q
```

The runbook also lists narrower command groups for Q2/Q3, Q4, and batch intake.

## Operational Notes

- The intended runtime is WSL/Linux.
- Student artifacts must be treated as untrusted inputs.
- Q4 uses subprocess isolation, but it is not a hardened multi-tenant sandbox.
- Non-functional Q4 submissions receive policy-driven zero outcomes.
- Review the generated JSON outputs, especially for low-confidence or failed cases.

## Reuse Notes

This repository has been cleaned for reuse:

- scratch folders such as `out/`, `sandbox/`, and temporary extraction artifacts are ignored
- public datasets remain in `data/`
- reusable reference fixtures remain in `reference/`

If you are adapting this project for a new assignment cycle, start by updating:

- `config/grading.toml`
- `specs/questions.toml`
- `specs/rubrics.toml`
- `GRADING_POLICY.md`
- `LLM_JUDGE_SPEC.md`

## Next Documents To Read

- `RUNBOOK.md` for setup, testing, and batch operations
- `GRADING_POLICY.md` for grading behavior and audit requirements
- `LLM_JUDGE_SPEC.md` for the Q2/Q3 judge contract
