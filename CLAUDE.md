# autotune-ollama

## Status reports

Run `./status.py` for a quick summary of an in-progress or completed run. It reads `details.jsonl` and `run.log` and shows:
- Per-model leaderboard (avg quality, record count, phase breakdown)
- Current activity (model/param being swept)
- Best configs with ≥4 samples
- Per-prompt quality breakdown for the leading model

## Findings / results

Run `./report.py` to get a full analytical summary of what the run has learned. Use this when answering "what have we learned so far?" It shows:
- Model ranking
- Best parameter configurations
- Parameter sensitivity (which params matter most, best values per model)
- Prompt difficulty ranking
- Inference failure summary
