# Notebooks: CV-Pilot on real data and real models (outputs saved from a real run)
`01_tutorial.ipynb`: doctor, record a run, score YOLO11n on COCO128 per class, pin a baseline, compare and gate a worse candidate, CI, kill and resume a training job.
`02_edge_cases.ipynb`: 37 checked cases (bad inputs, fail-closed gates, edited results, damaged or tampered checkpoints) ending in an expected-vs-actual table.
Data: COCO128 (128 photos; annotations CC BY 4.0, images keep their Flickr licences). Models: YOLO11n via `ultralytics` (AGPL-3.0, extra `cvpilot[ultralytics]`) and a small PyTorch classifier.
Both downloads (about 13 MB, SHA-256 pinned, checked before unpacking or loading) go to `$CVPILOT_NOTEBOOK_DATA` (default `<tmp>/cvpilot-notebooks`), never into the repository; the internet is needed once.
Run from the repository root; nothing is added to `pyproject.toml` or `uv.lock`:
`uv run --extra safetensors --extra remote --with jupyterlab --with ipykernel --with matplotlib --with ultralytics --with torch --with torchvision jupyter lab notebooks`
To re-execute and save the outputs: same command, with `jupyter nbconvert --to notebook --execute --inplace notebooks/01_tutorial.ipynb` instead of `jupyter lab notebooks`.
Each notebook starts in a fresh temporary folder with its own empty store, so it can be re-run from the top; on a laptop CPU 01 takes about 1 minute and 02 about 1.5 minutes. Notebook 02 reuses the detections that 01 cached, and computes them itself otherwise.
The notebooks are not part of the wheel or the sdist; `tests/notebooks` checks them without running anything.
