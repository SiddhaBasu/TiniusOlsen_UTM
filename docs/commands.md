# Command Reference

> Note: this command was originally written against `outlineandmeasure.py`, but that
> script has no `--use_fov`/`--fov_deg`/`--distance_in` CLI flags — the ArUco/FOV-calibration
> interface it describes actually belongs to `extract_outline.py` (now under `archive/`,
> see [docs/CODEBASE.md](CODEBASE.md)). Corrected below; run from the repo root.

```
python archive/extract_outline.py samples/black_sample.jpg --use_fov --fov_deg 60 --distance_in 12
```
