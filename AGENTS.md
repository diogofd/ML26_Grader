# AGENTS.md

This repository implements a grading pipeline for Assignment 1.

## Mission
Build a robust, auditable grader for Questions Q2, Q3, and Q4 only.

## Scope
In scope:
- Q2
- Q3
- Q4

Out of scope:
- Q1
- Q5
- plagiarism detection
- late penalties
- academic misconduct workflows
- UI/web app unless explicitly requested

## Source of truth
Always follow these documents first:
1. `GRADING_POLICY.md`
2. `LLM_JUDGE_SPEC.md`
3. any future rubric/spec files in the repo

If code and docs disagree, prefer the docs and surface the mismatch.

## Hard rules
- Q4 must remain fully deterministic.
- Do not use an LLM to score Q4.
- Q2 and Q3 must be graded from extracted evidence packets, not from raw whole-notebook freeform reading.
- Never invent grading criteria not present in the policy/spec documents.
- Never hard-code grade weights unless explicitly specified in the docs or config.
- Keep audit logging intact.
- Prefer small, reviewable changes.
- Prefer typed Python.
- Add or update tests for deterministic logic.

## Implementation priorities
Prioritize work in this order:
1. project scaffolding
2. submission validation
3. Q4 deterministic execution and scoring
4. Q2/Q3 evidence extraction
5. LLM judge interface conforming to the spec
6. reporting and feedback output

## Coding standards
- Use Python 3.11 or 3.12.
- Keep dependencies minimal.
- Use clear module boundaries.
- Prefer pure functions for deterministic grading logic.
- Validate structured outputs with schemas.
- Fail closed rather than guessing.

## Safety and trustworthiness
- Treat student artifacts as untrusted inputs.
- Do not assume pickle files or Python modules are safe.
- Do not silently swallow execution failures.
- Log failure reasons explicitly.

## When unsure
If requirements are ambiguous:
- make the smallest reasonable assumption,
- state it clearly in code comments or task summary,
- and avoid broad architectural changes.
