# Development Rules

This file is a pointer. The full operational contract lives in the root:

→ [`../rules_templet.md`](../rules_templet.md) (original template)
→ [`../CURSOR_RULES.md`](../CURSOR_RULES.md) (project-specific application)

## Quick reference

| Rule | Summary |
|---|---|
| Reproducibility | All results traceable to code + config + seed |
| Artifact-first | Export figures/tables/models; notebook output cells are not deliverables |
| Structure oracle | `context/structure.md` is always up to date |
| Run bundles | Every reportable experiment writes `artifacts/logs/runs/<run_id>/` |
| No magic numbers | All hyperparameters in `configs/*.yaml` |
| No absolute paths | Use `src/utils/paths.py` |
| No formation columns in test | See D002 in `DECISIONS.md` |
| Notebook done checklist | See `CURSOR_RULES.md` |

## When in doubt

1. Read `context/DECISIONS.md` for prior choices.
2. Read `context/ASSUMPTIONS.md` for constraints.
3. Check `context/STATUS.md` for what's next.
4. If making a new meaningful choice, add it to `DECISIONS.md`.
