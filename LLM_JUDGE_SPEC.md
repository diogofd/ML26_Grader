# LLM_JUDGE_SPEC.md

## Purpose

This document specifies how the LLM judging component evaluates Q2 and Q3.

The LLM judge must:
- receive a structured evidence packet,
- apply a strict rubric,
- score each subquestion separately,
- return one question-level result per call,
- generate one student-facing paragraph per subquestion,
- generate one composed student-facing paragraph for the whole question,
- generate short internal notes,
- and return confidence scores.

The LLM judge must not evaluate Q4.

## Scope

The LLM judge is responsible only for:
- Q2, consisting of:
  - Q2.1
  - Q2.2
- Q3, consisting of:
  - Q3.1
  - Q3.2

The LLM judge must not:
- execute student code,
- compute F1,
- curve grades,
- evaluate Q4,
- invent evidence,
- infer missing work from likely intent.

## Call structure

The system makes:
- one LLM call for Q2,
- one LLM call for Q3.

Each call must score all subquestions belonging to that question.

Therefore:
- the Q2 call must return scores for Q2.1 and Q2.2, plus an overall Q2 result,
- the Q3 call must return scores for Q3.1 and Q3.2, plus an overall Q3 result.

## Input contract

Each LLM call must include:

### 1. `question_id`
One of:
- `Q2`
- `Q3`

### 2. `question_text`
The official wording for the full question.

### 3. `subquestions`
A structured list of the subquestions for that question.

For example, for Q2:
- Q2.1 official wording
- Q2.2 official wording

### 4. `max_scores`
Official maximum scores for:
- the overall question
- each subquestion

### 5. `rubric_blocks`
A rubric block for each subquestion.

Each rubric block should include:
- required evidence,
- partial-credit guidance,
- common failure modes,
- score-band guidance,
- feedback guidance.

### 6. `evidence_packet`
A structured evidence packet assembled by the extraction layer.

It may contain:
- relevant markdown snippets,
- relevant code snippets,
- relevant output snippets,
- extracted signals,
- extraction warnings,
- detected models,
- detected metrics,
- preprocessing signals,
- tuning signals,
- comparison signals,
- business-justification signals.

### 7. `metadata`
Optional metadata such as:
- submission ID,
- notebook filename,
- rubric version,
- prompt version,
- extraction version.

## Core judging rules

The LLM must obey all of the following:

- grade only from explicit evidence in the evidence packet,
- do not use outside knowledge about what the student probably meant,
- do not reward work that is merely implied,
- do not invent missing steps,
- do not reward polished prose without evidential support,
- use the rubric exactly,
- be conservative with full marks,
- flag uncertainty rather than guessing,
- return valid structured JSON only.

## Output schema

The LLM must return exactly one JSON object per question call with the following structure:

```json
{
  "question_id": "Q2",
  "score": 0.0,
  "max_score": 0.0,
  "confidence": 0.0,
  "student_feedback_overall": "",
  "internal_notes_overall": "",
  "review_recommended": false,
  "review_reasons": [],
  "subquestions": {
    "Q2.1": {
      "score": 0.0,
      "max_score": 0.0,
      "confidence": 0.0,
      "student_feedback": "",
      "internal_notes": "",
      "evidence_used": [],
      "missing_requirements": []
    },
    "Q2.2": {
      "score": 0.0,
      "max_score": 0.0,
      "confidence": 0.0,
      "student_feedback": "",
      "internal_notes": "",
      "evidence_used": [],
      "missing_requirements": []
    }
  }
}
```

The same structure applies to Q3, replacing Q2.1/Q2.2 with Q3.1/Q3.2.

## Field definitions

### Top-level fields

#### `question_id`
Must be either:
- `Q2`
- `Q3`

#### `score`
Overall score for the question.

Constraints:
- `0 <= score <= max_score`
- the intended value is the sum of the subquestion scores

The implementation must treat the sum of subquestion scores as the canonical source of truth.

#### `max_score`
Overall maximum score for the question.

Constraint:
- must equal the sum of the subquestion maximum scores

#### `confidence`
Question-level self-assessed grading confidence on a 0 to 10 scale.

Interpretation:
- `0` = extremely unreliable judgment
- `10` = extremely reliable judgment

This is confidence in the correctness of the full-question grading decision.

This is the confidence value used for gating auto-pass.

#### `student_feedback_overall`
A composed student-facing paragraph for the whole question.

This should synthesize:
- the main strengths across the subquestions,
- the main weaknesses across the subquestions,
- the overall reason for the final score.

It must be a synthesis of the subquestion-level judgments and must not introduce unsupported claims.

It should read naturally and not feel like a pasted list.

Target length:
- about 100 to 220 words

#### `internal_notes_overall`
A short grader-facing summary for the question as a whole.

This should:
- summarize the strongest evidence,
- summarize the main scoring gaps,
- mention overall ambiguity if present.

Target length:
- about 20 to 80 words

#### `review_recommended`
Boolean.

Must be `true` if:
- question-level confidence < 8.5,
- any subquestion is too ambiguous for reliable scoring,
- the packet is incomplete for a reliable question-level judgment,
- the score depends on uncertain interpretation.

This field is advisory to the grading pipeline.
The pipeline remains the final authority on:
- hard fail-closed routing,
- soft review routing,
- and any configured soft auto-pass exception.

#### `review_reasons`
A list of short machine-readable reasons explaining why review is recommended or why the result may require special attention.

Examples:
- `"question_confidence_below_threshold"`
- `"subquestion_ambiguity"`
- `"missing_required_evidence"`
- `"fragmented_evidence_packet"`
- `"score_consistency_issue"`

If `review_recommended` is `false`, this list should usually be empty.

### Subquestion fields

Each subquestion object must contain:

#### `score`
Numeric score for that subquestion.

Constraint:
- `0 <= score <= max_score`

#### `max_score`
Official maximum score for that subquestion.

#### `confidence`
Subquestion-level grading confidence on a 0 to 10 scale.

This is stored for auditing and moderation, but auto-pass gating is determined at question level.

#### `student_feedback`
One polished student-facing paragraph for that subquestion.

Requirements:
- constructive,
- concise,
- grounded in the actual evidence,
- readable and professional,
- no internal grader jargon,
- no mention of thresholds, confidence, or automation.

Target length:
- about 60 to 140 words

#### `internal_notes`
Short grader-facing note for that subquestion.

Requirements:
- concise,
- operational,
- evidence-based,
- may mention ambiguity directly.

Target length:
- about 15 to 60 words

#### `evidence_used`
List of short evidence bullets actually used to determine the score.

Examples:
- `"Two distinct models are explicitly trained"`
- `"Train/test split is shown in code"`
- `"Metric choice is tied to false negatives and business cost"`

#### `missing_requirements`
List of rubric elements not explicitly supported by the evidence.

Examples:
- `"No clear hyperparameter tuning evidence"`
- `"Business justification for model choice is weak"`
- `"Comparison is descriptive but not tied to deployment decision"`

## Confidence calibration rules

The LLM must produce:
- one confidence score per subquestion,
- one overall confidence score per question.

### Subquestion confidence
Subquestion confidence should be high only when:
- the evidence for that subquestion is clear,
- the rubric match is straightforward,
- the score is well supported by explicit evidence.

### Question-level confidence
Question-level confidence should reflect the reliability of the full question judgment as a whole.

It should be reduced when:
- one or more subquestions are ambiguous,
- evidence is fragmented across the notebook,
- extraction warnings suggest missing context,
- the distinction between adjacent score bands is uncertain,
- the overall score depends on a borderline interpretation.

Suggested calibration:
- `9.0–10.0`: very clear evidence, low ambiguity
- `8.5–8.9`: acceptable for auto-pass
- `8.0–8.4`: plausible and usually review-worthy, but may qualify for a configured soft auto-pass if the pipeline determines the case is strong, coherent, and soft-warning-only
- `7.0–7.9`: plausible but should normally be reviewed
- `<7.0`: unreliable or strongly ambiguous

The model must not inflate confidence just because the answer sounds reasonable.

## Student feedback style rules

### Subquestion feedback
Each subquestion must have its own paragraph.

That paragraph should:
- lead with substantive strengths when present,
- identify the main scoring gaps,
- sound professional and constructive,
- remain specific without being verbose,
- refer to the student’s work, not to the grading system.

It must not:
- mention confidence,
- mention thresholds,
- mention internal review routing,
- mention “the model,” “the LLM,” or “the automated grader.”

Bad example:
- `"The automated grader could not detect enough evidence of tuning."`

Good example:
- `"Your preprocessing and model training steps are clearly shown, but the tuning process and the rationale for your final model choice needed to be explained more explicitly."`

### Overall feedback
The overall feedback paragraph should synthesize the subquestion feedback into one readable paragraph.

It should not merely concatenate the two subquestion paragraphs.

It should:
- reflect the overall balance of strengths and weaknesses,
- explain the overall score coherently,
- remain student-facing and polished,
- stay grounded in the same evidence base as the subquestion feedback.

It must not introduce new unsupported claims.

## Internal notes style rules

### Subquestion internal notes
Should be:
- short,
- evidence-based,
- operational,
- explicit about the main gap.

Example:
- `"Clear split, preprocessing, and two-model workflow. Tuning evidence weak. Confidence 8.6."`

### Overall internal notes
Should:
- summarize the question-level judgment,
- mention the strongest evidence,
- mention the principal weakness,
- mention ambiguity if present.

## Prompt contract

The system prompt must instruct the model to act as a strict rubric-based grader.

It must explicitly require:
- use only the rubric and evidence packet,
- no inference beyond explicit evidence,
- no outside assumptions,
- return JSON only,
- set `review_recommended=true` whenever question-level confidence < 8.5 or ambiguity remains.

The prompt may still use `8.5` as the default review threshold.
Any configured soft auto-pass exception is enforced by pipeline policy after structured validation, not by changing the output schema.

The user prompt must include:
- question ID,
- question text,
- subquestion texts,
- max scores,
- rubric blocks,
- evidence packet,
- extraction warnings if any.

## Score consistency rules

The LLM should return both:
- subquestion scores,
- and an overall question score.

However, the implementation must treat the sum of subquestion scores as canonical.

Post-processing must:
- validate whether the reported overall score matches the subquestion sum,
- log any inconsistency,
- either repair deterministically by setting the overall score to the subquestion sum, or route the result to review, depending on implementation policy.

If the inconsistency is material or accompanies other ambiguity, it should contribute to:
- `review_recommended = true`
- and inclusion of `"score_consistency_issue"` in `review_reasons`.

## Error handling

If the model returns invalid JSON:
- retry once with a repair prompt,
- if still invalid, fail closed and route to review.

If required fields are missing:
- attempt one schema repair,
- otherwise fail closed and route to review.

If any score is out of bounds:
- reject the output,
- log the failure,
- route to review.

If the question ID or subquestion IDs do not match the request:
- reject the output,
- route to review.

If the total score does not equal the sum of subquestion scores:
- handle it according to the score consistency rules above,
- log the event,
- route to review if needed.

## Determinism and settings

The implementation should use low-variance generation settings.

Recommended behavior:
- low temperature,
- schema validation after generation,
- explicit version logging.

The implementation must log:
- provider,
- model name,
- model version if available,
- prompt version,
- rubric version,
- extraction version.

## Acceptance rule

A question result may be auto-accepted only if:
- question-level `confidence >= 8.5`,
- `review_recommended == false`,
- JSON output passes schema validation,
- all scores are within bounds,
- and the final stored overall score is consistent with the sum of subquestion scores.

Pipeline exception:
- if enabled by configuration, the grading pipeline may soft-auto-pass a strong and internally coherent question result with `8.0 <= confidence < 8.5`,
- but only when no hard-blocking reasons remain,
- only when the stored score is consistent,
- and only when the remaining concerns are soft-review reasons rather than fail-closed conditions.

Subquestion confidence scores are stored for auditability, but auto-pass gating is determined at question level.

## Non-goals

This document does not define:
- how evidence is extracted,
- how Q2/Q3 scores are aggregated into course grades,
- how Q4 is executed,
- how leaderboard rank is computed,
- how final grades are published.

Those belong in separate specifications.
