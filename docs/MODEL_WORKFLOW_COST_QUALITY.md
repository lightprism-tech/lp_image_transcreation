# Model Workflow for Best Output with Lower Cost

## Goal

This report maps directly to the current 3-stage pipeline:

1. Stage 1: Perception (image to structured text/details)
2. Stage 2: Reasoning (edit plan generation)
3. Stage 3: Realization (image editing)

The target is to keep cost low while preserving output quality.

---

## Current Pipeline Fit

Based on current code and configuration:

- Stage 1 already runs mostly local models (`YOLO`, `BLIP`, `CLIP`, `PaddleOCR`), so recurring API cost is low.
- Stage 2 supports `groq` or `openai` via `LLM_PROVIDER`.
- Stage 3 uses an inpainter backend and can run with inpainting plus quality gates and fallbacks.

This means your main recurring cloud cost is Stage 2 (LLM) and optional Stage 3 external image API usage.

---

## Recommended Operating Modes

## 1) Default Production Mode (Best cost-to-quality)

Use this as the daily default.

- **Stage 1 (Perception):** use hybrid local-first perception.
  - Why: local is cheap, but hard images need cloud-level understanding to avoid missed details.
- **Stage 2 (Reasoning):** use `groq` with a fast low-cost model first.
  - Suggested: `llama-3.1-8b-instant` for first pass.
- **Stage 3 (Realization):** keep local inpaint/edit flow enabled with current quality gates.
  - Why: avoids per-image generation API costs when local output is acceptable.

When to use:

- bulk processing
- internal QA runs
- fast iteration while tuning prompts/config

Expected result:

- lowest operational cost
- good baseline quality
- fast turnaround

---

## 2) Smart Escalation Mode (Best quality under budget control)

Run cheap first, then escalate only hard cases.

### Escalation policy

1. Run all images in Default Production Mode.
2. Escalate only if one or more conditions fail:
   - Stage 2 plan is weak (very low confidence / empty actionable transforms where action is expected)
   - Stage 3 quality gates fail repeatedly
   - human reviewer flags cultural mismatch or semantic drift
3. Re-run only failed samples with stronger models/settings.

### Escalation path

- **Stage 2 escalation:** move from low-cost model to stronger reasoning model (OpenAI or stronger Groq model if available to your account).
- **Stage 3 escalation:** move selected failed cases to premium image edit backend (for example provider-hosted FLUX or GPT-image style editing endpoint) while keeping the same edit-plan constraints.

Expected result:

- near-premium output quality on hard examples
- much lower total cost than always using premium models

---

## 3) Premium Quality Mode (Use for demos and critical outputs)

- Strongest Stage 2 reasoning model for all runs.
- Premium Stage 3 image edit backend for all runs.

Use only when:

- external client delivery
- showcase/demo outputs
- high-value low-volume assets

Trade-off:

- highest quality consistency
- highest cost

---

## Stage-by-Stage Model Strategy

## Stage 1: Image to full detail text (Perception)

Current best approach for your project:

- keep local stack as primary:
  - YOLO for objects
  - BLIP for dense captions
  - OCR for text
  - CLIP for global semantic support

- add a cloud vision verifier on low-confidence samples:
  - run a stronger multimodal model only for hard images
  - merge local + cloud outputs into a final Stage-1 JSON for Stage 2

### Recommended hybrid flow

1. Run local Stage-1 extraction.
2. Compute Stage-1 confidence signals.
3. If confidence is good, continue to Stage 2 directly.
4. If confidence is low, call cloud vision verification and merge results.
5. Save merged Stage-1 JSON and continue to Stage 2.

### Stage-1 escalation triggers (practical thresholds)

Escalate to cloud verifier when any condition is true:

- detected objects fewer than `3` for non-document images
- OCR text regions detected but low text confidence trend (for example median confidence below `0.65`)
- image type is `infographic`, `poster`, or `document` with dense layout
- high overlap/noise in detections (many near-duplicate boxes)
- previous run metrics show Stage-3 failure or weak cultural/object scores for similar inputs

These thresholds should be tuned using your `*_run_metrics.json` logs.

Cost optimization tips:

- keep detector thresholds tuned to reduce noisy objects
- process at practical resolution before heavy runs
- cache outputs and reuse stage JSON (`--no-cache` only when needed)

Quality tips:

- maintain OCR region metadata for Stage 2/3
- keep image type classification active for infographic/document routing
- keep source trace metadata showing which fields came from local vs cloud verifier

---

## Stage 2: Edit-plan generation (Reasoning LLM)

Recommended order:

1. **Primary (low cost):** `groq` + `llama-3.1-8b-instant`
2. **Escalation (hard samples):** stronger reasoning model (OpenAI or higher-tier Groq model available in your environment)

Why this works:

- Stage 2 is structured JSON reasoning, not long-form generation.
- most easy/medium cases do not need expensive LLMs.
- hard cases can be selectively escalated after Stage-1 hybrid verification.

Practical controls:

- keep strict JSON output enforcement
- log and monitor empty or low-confidence plans
- escalate only flagged samples

---

## Stage 3: Image editing (Realization)

Recommended order:

1. Local inpaint/edit backend first (current engine + quality gates)
2. Premium external editor only for failed/critical outputs

Why:

- local first avoids per-image API charges
- quality gates already exist to block weak outputs
- fallback chain guarantees output even if full inpaint is unavailable

Quality control:

- keep text quality gate enabled for translated text regions
- keep composite validation threshold tuned for target quality
- use one retry pass before escalation

---

## Suggested Environment Profiles

Use separate `.env` profiles for repeatable operation.

## Profile A: low_cost_default

- `LLM_PROVIDER=groq`
- `LLM_MODEL=llama-3.1-8b-instant`
- local realization backend active
- stage cache enabled

## Profile B: balanced_escalation

- same as Profile A by default
- escalation routing enabled in orchestration logic/process
- only flagged items move to stronger Stage 2/Stage 3

## Profile C: premium_delivery

- strongest available reasoning model
- premium image edit backend for all outputs
- stricter validation thresholds

---

## Decision Matrix

| Scenario | Stage 1 | Stage 2 | Stage 3 | Recommended Mode |
|---|---|---|---|---|
| Large batch, internal use | hybrid local-first | low-cost | local | Default Production |
| Mixed quality incoming images | hybrid local-first + cloud verify on low confidence | low-cost + escalate failures | local + escalate failures | Smart Escalation |
| Client demo/final assets | hybrid with strict verification | strongest | premium | Premium Quality |

---

## Implementation Roadmap (No major refactor)

1. Keep local Stage-1 as default.
2. Add Stage-1 confidence scoring and cloud verification triggers.
3. Merge local + cloud Stage-1 outputs into one canonical JSON.
4. Keep Stage-2 low-cost default and escalate only flagged cases.
5. Route only failed images to premium Stage-3 models.
6. Track per-run metrics (`*_run_metrics.json`) to tune thresholds monthly.

This gives the best practical mix: strong quality, predictable cost, and minimal architecture change.

---

## Final Recommendation

For your current codebase, the best strategy is:

- use Stage-1 hybrid local-first perception with cloud verification on low-confidence images
- run Stage 2 on low-cost Groq model by default
- run Stage 3 local by default
- escalate only difficult or business-critical cases to stronger/premium models

This is the most cost-efficient path that still achieves high output quality.
