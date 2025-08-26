<!-- Progress Checklist: Update the boxes as you complete tasks -->
<!-- [ ] = pending, [-] = in progress, [x] = done -->

Overall Progress
- [x] Phase 0 — Baseline and Validation
- [ ] Phase 1 — Wire Action to Anthropic API (LLM in the loop)
- [ ] Phase 2 — Label Set Definition and Governance
- [ ] Phase 3 — Dataset Creation for Fine-tuning
- [ ] Phase 4 — Training and Evaluation
- [ ] Phase 5 — Hosting and Inference API
- [ ] Phase 6 — Productionization and Safeguards
- [ ] Phase 7 — Documentation and Maintenance

# Auto Issue Labeler: End-to-End Plan

This plan describes the steps to implement an automated issue labeling system that starts with a GitHub Action calling an LLM (Anthropic) and evolves toward a fine-tuned model hosted behind a stable inference API.

References (for current repo):
- Existing workflow: [`.github/workflows/issue-labeler.yml`](.github/workflows/issue-labeler.yml)
- Current labeling script: [`python.def main()`](scripts/label_issue.py:121)

Guiding principles:
- Ship in small increments; each phase has testable deliverables.
- Maintain reproducibility and traceability of prompts, datasets, and models.
- Keep a safe fallback path (comment-only recommendations before auto-apply; confidence thresholds).

--------------------------------------------------------------------------------

Phase 0 — Baseline and Validation
Goal: Confirm the existing GitHub Action triggers and minimal labeling loop.

Steps:
- Review the current workflow trigger: issues.opened
- Validate permissions: contents: read, issues: write (correct)
- Confirm script runs in Actions and can apply labels using placeholder logic in [`python.def determine_labels()`](scripts/label_issue.py:38)
- Add basic logging to verify payload handling
- Acceptance criteria:
  - On new issue open, job runs and logs show parsed title/body; if simple keywords present, labels are applied successfully.
  - No unhandled exceptions in typical issue payloads.

Deliverables:
- Working baseline with keyword heuristic labels (already present).

--------------------------------------------------------------------------------

Phase 1 — Wire Action to Anthropic API (LLM in the loop)
Goal: Replace placeholder labeling with an LLM recommendation flow.

Design:
- Prompt: Provide issue title, body, optional repo context, and the allowed label set; ask for JSON output with labels and confidences.
- Output schema (JSON):
  {
    "labels": [{"name": "bug", "confidence": 0.87}, ...],
    "rationale": "short explanation",
    "version": "v1"
  }
- Safety: Timeouts, retries, and abstention if low confidence.

Steps:
- Add secret ANTHROPIC_API_KEY in repo settings
- Extend [`python.def main()`](scripts/label_issue.py:121) flow:
  - Build prompt
  - Call Anthropic API (Claude 3.5 Sonnet recommended)
  - Parse/validate JSON response
  - Map to allowed labels; apply above threshold
- Add dry-run mode (env LABELER_DRY_RUN=true) that posts a comment recommending labels instead of applying
- Add threshold envs: LABELER_MIN_CONF=0.6; LABELER_MAX_LABELS=3
- Support actions for "opened" and optionally "edited" to re-evaluate labels
- Rate-limit handling: single retry with backoff; log and gracefully exit on persistent failure

Acceptance criteria:
- For diverse test issues, action posts recommended labels with confidences in dry-run, and applies labels when dry-run disabled if confidences exceed threshold
- Errors fail the job with clear logs but do not spam API

Deliverables:
- Updated workflow to include ANTHROPIC_API_KEY
- Updated script with Anthropic client call and JSON schema validation

--------------------------------------------------------------------------------

Phase 2 — Label Set Definition and Governance
Goal: Curate a clear, minimal-overlap label taxonomy for automation and future fine-tuning.

Steps:
- Draft initial label set (example):
  - bug, enhancement, documentation, question, ux, performance, test, security, refactor, ci
- Provide label descriptions, inclusion criteria, and examples
- Store machine-readable label config with metadata:
  - .github/labels.yml with fields: name, description, color, aliases, examples, deprecated
- Create/migrate labels in repo:
  - One-off script or gh cli to ensure label existence and descriptions
- Governance:
  - Rules to add/deprecate labels and update the config
  - Keep number manageable (10–20) and avoid redundancy

Acceptance criteria:
- Labels exist in the repo with clear descriptions
- The LLM prompt uses the exact allowed set and descriptions from the config

Deliverables:
- .github/labels.yml
- Script or instructions to sync labels

--------------------------------------------------------------------------------

Phase 3 — Dataset Creation for Fine-tuning
Goal: Build a high-quality dataset to train a model for automated labeling.

Data model:
- JSONL entries with:
  - id, title, body, labels (multi-label list), annotator, created_at
  - split: train/val/test
  - source: manual/LLM_bootstrap/heuristic
  - rationale (optional), comments (optional)

Steps:
- Data sourcing:
  - Past repo issues and labels (if any)
  - Manual triage sessions to label 200–1,000 examples
  - LLM-assisted bootstrapping with human verification to scale
- Annotation protocol:
  - Multi-label allowed; cap at 3 labels per issue
  - Clear decision guide; favor abstain/other over incorrect labels
- Quality control:
  - Inter-annotator agreement (Cohen’s kappa)
  - Spot checks per label
  - Balance and coverage review
- Versioning:
  - Store datasets in data/issues/*.jsonl
  - Consider DVC or git-lfs if large
  - Dataset card with stats

Acceptance criteria:
- At least 1,000 high-quality examples with balanced label distribution
- train/val/test stratified by label with no leakage by issue id

Deliverables:
- data/issues/train.jsonl, val.jsonl, test.jsonl
- docs/datasets/ISSUE_LABELS.md (dataset card)

--------------------------------------------------------------------------------

Phase 4 — Training and Evaluation
Goal: Establish a training pipeline and baseline performance.

Approach options:
- Vendor inference (no fine-tune): prompt engineering and few-shot exemplars to reach baseline
- Open model fine-tune (recommended for control/cost):
  - Model: Llama 3 or Mistral; use PEFT/LoRA
  - Input: title + body (+ optional repo metadata)
  - Output: multi-label classification head or instruction format with labels

Metrics:
- Micro/macro F1, per-label precision/recall, confusion matrix
- Throughput and latency for inference

Steps:
- Implement training script: training/train.py with:
  - Data loader for JSONL
  - Tokenization and formatting (instruction style if using generative)
  - LoRA config
  - Early stopping on val F1
- Evaluation:
  - Compute metrics and save report with best threshold per label
- Model selection criteria:
  - Macro-F1 ≥ baseline LLM prompt performance
  - Stable per-label precision ≥ specified thresholds

Acceptance criteria:
- Reproducible training runs producing a model and report
- Documented hyperparameters and seed

Deliverables:
- [`python.def train()`](training/train.py:1)
- reports/eval_{timestamp}.json, confusion matrices
- saved model artifacts (local or remote)

--------------------------------------------------------------------------------

Phase 5 — Hosting and Inference API
Goal: Provide a stable inference endpoint.

Options:
- Continue vendor LLM (Anthropic): zero hosting; call directly in action
- Host fine-tuned open model:
  - Hugging Face Inference Endpoints / Replicate (managed)
  - Self-host with vLLM or TGI behind a small FastAPI
- API contract:
  - POST /predict
    - Input: {title, body, allowed_labels, top_k, thresholds}
    - Output: {labels: [{name, confidence}], rationale, version}

Steps:
- Package model server
- Add autoscaling or concurrency controls
- Observability: basic request metrics and error logs

Acceptance criteria:
- Endpoint returns predictions within SLA (p95 < 2s if feasible)
- Compatible with action client

Deliverables:
- server/ (FastAPI) or deployment config for managed hosting
- Client integration in action

--------------------------------------------------------------------------------

Phase 6 — Productionization and Safeguards
Goal: Make it safe and maintainable.

Mechanisms:
- Confidence thresholds and abstention label
- Canary rollout:
  - Stage 1: comment-only recommendations
  - Stage 2: auto-apply when all recommended labels ≥ threshold
  - Stage 3: auto-apply with cap and human-in-the-loop override
- Slash commands:
  - /label auto — force re-run
  - /label off — prevent auto-label on this issue
- Auditability:
  - Post a brief comment including version, labels, confidences
  - Store structured log artifacts if needed
- Monitoring:
  - Weekly accuracy samples (human review 20 random issues)
  - Drift detection via label distribution changes

Acceptance criteria:
- No unexpected mass relabeling
- Clear on/off controls
- Observed accuracy trend maintained or improved

Deliverables:
- Action logic for canary modes and slash commands
- Simple monitoring script or GitHub Insights queries

--------------------------------------------------------------------------------

Phase 7 — Documentation and Maintenance
Goal: Ensure clarity and continuity.

Docs:
- README: how the action works, secrets required, config knobs, dry-run
- Ops runbook: rotating keys, handling outages, retry strategies
- Contribution guide: updating labels.yml, adding data, re-training
- Changelog: prompt changes, dataset versions, model versions

Acceptance criteria:
- A new contributor can run end-to-end (dry-run) in under an hour
- Clear upgrade path from Anthropic-only to hosted model

Deliverables:
- README updates
- docs/ops/runbook.md
- docs/CONTRIBUTING.md

--------------------------------------------------------------------------------

Immediate Next Actions (for this repo)
1) Phase 1 - Anthropic wiring (LLM call)
- Add secret ANTHROPIC_API_KEY
- Update workflow to pass ANTHROPIC_API_KEY to the job
- Extend [`python.def main()`](scripts/label_issue.py:121) to:
  - Build prompt from issue title/body and allowed labels (temporary hardcoded set: bug, enhancement, documentation)
  - Call Anthropic API with timeout
  - Parse labels, filter against allowed set, threshold at 0.6, cap 3
  - Dry-run initial (comment recommendations)

2) Phase 2 - Label set
- Draft .github/labels.yml
- Provide script/instructions to sync labels and descriptions

3) Phase 3 - Dataset
- Add data schema and empty skeleton files
- Write dataset card template with annotation guidance

Change log coordination
- Record all prompt/schema changes in a PROMPTS.md
- Version outputs with a version field in the JSON response for traceability

Appendix — Risk register
- Token/secret leakage: mitigate via GitHub secrets and minimal logs
- Over-labeling: cap labels, use thresholds, canary comment-only
- Rate limits/vendor downtime: retries and graceful degradation
- Dataset bias: stratified sampling, inter-annotator checks, periodic reviews