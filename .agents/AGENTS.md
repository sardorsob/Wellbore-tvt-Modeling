# Agent Rules

These rules apply to AI-assisted work in this repository.

## Workflow

- Use the project context files as the local source of truth, especially `context/structure.md`, `context/STATUS.md`, and `CURSOR_RULES.md`.
- Use `sardorsob/sardors-agentic-workflows` as the workflow reference for structured planning, execution, documentation, and delivery.
- If a local copy of that workflow exists under `context/_external/`, treat it as reference-only material and never commit it.
- Keep notebook work artifact-first: exported figures/tables/models plus updates to `context/structure.md`.
- Record meaningful decisions in `context/DECISIONS.md` and assumptions in `context/ASSUMPTIONS.md`.
- Delete one-off notebook/script generators after their output is produced unless they are intentionally reusable project tooling.
- For Bayesian work, record the generative story, priors, likelihood choices, and prior/posterior predictive assumptions before treating results as reportable.

## Git

- Commit regularly when a coherent unit of work is complete.
- Use a clear imperative commit title and a short descriptive body when useful.
- Never add AI co-author trailers or self-attribution to commits.
- Do not commit raw Kaggle data, credentials, local workflow reference copies, or generated heavyweight model binaries.

## Kaggle

- Prefer submissions produced by reproducible notebooks or scripts.
- Validate `submission.csv` shape, id ordering, finite predictions, and `sample_submission.csv` alignment before any Kaggle upload.
