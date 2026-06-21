# Kaggle Rules Summary

## Known From Local Context

- Kaggle Notebook runtime limit is 9 hours.
- Kaggle Notebook has no internet access during committed competition runs.
- External data is allowed only when publicly available and compliant with competition rules.
- Final submission must be produced by a committed Kaggle Notebook.
- Kaggle credentials and API keys must never be committed.

## Must Verify On Kaggle Before First Submission

- Daily submission limit.
- Exact external data policy wording.
- Teaming, sharing, and code/notebook reuse restrictions.
- Whether model artifacts can be uploaded as notebook inputs.
- Any competition-specific notebook or hardware restrictions.

## Operating Rules

- Do not tune only to the public leaderboard.
- Freeze local CV strategy before repeated submissions.
- Log every submission in `kaggle/SUBMISSIONS.md`.
- Keep every submitted file reproducible from code, config, and a Git SHA.
- Record public/private leaderboard mismatch risks in `kaggle/LEADERBOARD_NOTES.md`.

