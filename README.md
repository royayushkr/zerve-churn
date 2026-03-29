# Zerve Churn Streamlit App

This repository contains the deployment-ready Streamlit app and precomputed artifacts for the Zerve success/churn demo.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Included

- `app.py`: Streamlit entrypoint
- `src/`: lightweight inference helpers used by the app
- `artifacts/`: precomputed model files and runtime data needed by the app

No training is required to run this app.
