# GRADING_POLICY.md

## Purpose

This document defines the grading policy for the automated first-line grader for Assignment 1.

The grader evaluates only:
- Q2
- Q3
- Q4

The system separates:
- deterministic grading for Q4, and
- rubric-based LLM judging for Q2 and Q3.

The goal is to produce auditable provisional grades, polished student-facing feedback, and a review queue for low-confidence cases.

## Scope

In scope:
- Q2
- Q3
- Q4
- student-facing feedback for Q2 and Q3
- internal grader notes for Q2 and Q3
- deterministic leaderboard generation for Q4

Out of scope:
- Q1
- Q5
- plagiarism detection
- late penalties
- academic misconduct workflows
- appeals workflows
- final publication UI
- final Q4 rank-to-grade mapping

## Source of truth

The official assignment brief is the source of truth for:
- question wording
- point weights
- required deliverables
- Q4 evaluation mechanics

The implementation must load official point values from configuration.

Required configuration keys:
- `Q2_1_MAX_POINTS`
- `Q2_2_MAX_POINTS`
- `Q3_1_MAX_POINTS`
- `Q3_2_MAX_POINTS`
- `Q4_MAX_POINTS`

Derived values:
- `Q2_MAX_POINTS = Q2_1_MAX_POINTS + Q2_2_MAX_POINTS`
- `Q3_MAX_POINTS = Q3_1_MAX_POINTS + Q3_2_MAX_POINTS`

These values must not be hard-coded unless explicitly copied from the brief into configuration.

## Grading units

The system grades the following units:

### LLM-graded units
- `Q2.1`
- `Q2.2`
- `Q3.1`
- `Q3.2`

### Deterministic unit
- `Q4`

The system must store:
- subquestion scores,
- aggregated question scores,
- and any review flags.

## High-level grading model

### Q2 and Q3
Q2 and Q3 are graded by an LLM judge using:
- a strict rubric,
- an extracted evidence packet,
- and a structured output schema.

The judge scores each subquestion separately.

The system then aggregates:
- `Q2 = Q2.1 + Q2.2`
- `Q3 = Q3.1 + Q3.2`

### Q4
Q4 is graded deterministically.

The system must:
- validate the submission artifacts needed for execution,
- run the submitted model or pipeline on the external evaluation dataset,
- validate the predictions,
- compute F1 score,
- and generate a leaderboard rank.

At this stage, Q4 produces:
- `f1_score`
- `rank`
- `leaderboard_status`

The final mapping from leaderboard rank to grade is intentionally deferred.

## Confidence rule for Q2 and Q3

The LLM must return a confidence score on a 0 to 10 scale for each graded subquestion.

Standard auto-accept rule:
- a question result may be accepted automatically when question-level `confidence >= 8.5`
- and no hard-blocking or disallowed soft-review reasons remain after pipeline review-policy assessment

Soft auto-pass exception:
- the pipeline may auto-accept a question result with question-level confidence below `8.5`
- only if the configured soft auto-pass policy is enabled,
- only for strong, coherent, non-failed cases,
- and only when the remaining concerns are soft-review reasons rather than hard-blocking failures

Hard-blocking rule:
- extraction failure, invalid structured judge output, missing rubric/config/spec data, score consistency issues, and comparable hard-blocking failures must still fail closed and route to review

Subquestion confidence scores remain stored for audit, but final auto-pass routing is determined at the question level by the grading pipeline.

## Review policy

A submission or subquestion must enter the review queue if any of the following occurs:

- question-level confidence below 8.5 and the soft auto-pass policy does not apply
- invalid or unparsable LLM output
- missing required output fields
- evidence extraction failure
- ambiguous notebook structure
- contradictory evidence
- deterministic checks contradict narrative claims in a way that affects grading
- unsupported artifact structure
- manual override requested by instructor
- Q4 execution failure
- malformed Q4 predictions

Review means human moderation, not automatic rejection.

The pipeline distinguishes:
- hard-blocking cases, which fail closed and require review,
- soft-review cases, which remain reviewable but may be auto-passed if the configured soft auto-pass criteria are met,
- and strong auto-pass cases, which can be accepted without review despite light non-blocking warnings.

## Q2 grading policy

Q2 must be graded at subquestion level.

### Q2.1
The LLM should assess whether the student correctly formulates the machine learning task required by the brief.

The judgment must rely only on explicit evidence in the evidence packet.

### Q2.2
The LLM should assess whether the student develops the required predictive models and supports that work with an appropriate technical workflow.

Depending on the rubric, relevant evidence may include:
- model development
- preprocessing
- feature engineering
- train/test or validation split
- hyperparameter tuning
- technical justification
- business justification

The LLM must not award marks for work that is only implied.

## Q3 grading policy

Q3 must be graded at subquestion level.

### Q3.1
The LLM should assess whether the student selects and justifies evaluation metrics appropriately.

Depending on the rubric, relevant evidence may include:
- metric choice
- relation to business risk
- relation to decision-making constraints
- relation to regulatory or operational concerns

### Q3.2
The LLM should assess whether the student compares models appropriately and recommends one for deployment with a coherent justification.

The LLM must judge only from explicit evidence.

## Q4 grading policy

Q4 must not use an LLM.

Q4 is evaluated deterministically.

The system must:
- validate artifacts required for execution,
- run inference on the external evaluation dataset,
- verify prediction validity,
- compute F1 score,
- and compute leaderboard rank among valid submissions.

For now, Q4 output is leaderboard-oriented only.

The system must store:
- `f1_score`
- `rank`
- `leaderboard_status`
- execution status
- failure reason if applicable

## Failed-run policy for Q4

A Q4 run is considered failed if any of the following occurs:

- required execution artifacts are missing
- the model or pipeline cannot be loaded
- required modules cannot be imported
- inference crashes
- prediction count does not match input rows
- predictions are not binary in the required format
- predictions are empty
- runtime exceeds timeout
- submission violates sandbox constraints

A failed run must be recorded with:
- failure category
- failure message
- execution logs if available

## Student-facing feedback policy

The system must prioritize polished student-facing feedback for:
- Q2.1
- Q2.2
- Q3.1
- Q3.2

Student-facing feedback must be:
- constructive
- concise
- professional
- grounded in the student’s submission
- understandable without internal grader jargon

The system should also store:
- a shorter internal grader note for each subquestion

The student-facing feedback is the primary output.
The internal note is secondary but required.

## Aggregation policy

The system must store both:
- subquestion-level results
- question-level aggregates

Aggregation rules:
- `Q2 = Q2.1 + Q2.2`
- `Q3 = Q3.1 + Q3.2`

If one or more subquestions are routed to review:
- the aggregate may still be computed provisionally,
- but it must be marked as non-final until review is completed.

If a question is soft-auto-passed:
- the aggregate may be stored as scored,
- the soft-review reason codes must still remain available in the audit trail,
- and the result must record that soft auto-pass was applied.

## Auditability requirements

Every grading result must be traceable.

The system must persist:
- evidence packet version
- rubric version
- prompt version
- model name/version
- raw LLM output
- parsed LLM output
- confidence score
- review flag
- student-facing feedback
- internal notes
- Q4 execution logs
- Q4 F1 score
- Q4 leaderboard rank
- timestamps for all grading steps

No final or provisional result should exist without an audit trail.

## Non-goals

This document does not define:
- the exact rubric text
- the evidence extraction algorithm
- the LLM prompt contents
- the database schema
- the final Q4 curve-to-grade mapping
- publication logic for the leaderboard

Those belong in separate specification documents.
