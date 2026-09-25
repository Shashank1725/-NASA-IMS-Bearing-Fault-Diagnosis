# BearingIQ web application

This project now uses a FastAPI backend and a standalone HTML/CSS/JavaScript frontend.

## Run locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000.

## Dynamic evaluation

- Every uploaded CSV is analyzed from scratch.
- Predictions are produced only when the trained feature columns are present.
- Accuracy, precision, recall, F1, and the confusion matrix are generated only when a recognized label column (`label`, `target`, `class`, `fault`, `failure`, `bearing_state`, or `status`) is present.
- Unlabelled datasets show live risk and prediction distributions but do not receive fabricated evaluation scores.
- The Three.js background uses generated particles only; no 3D object files are imported.
