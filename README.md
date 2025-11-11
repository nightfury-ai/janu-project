# Job Fraud Predictor (Flask)

This small Flask app loads a saved LightGBM model (`lightgbm_job_scam_model.pkl`) and exposes a web form and an API to predict whether a job posting is fraudulent.

Files added/updated:
- `flask_app.py` — Flask app that loads the model and serves `/` and `/api/predict`.
- `templates/index.html` — Jinja2 template with a form and dataset preview.
- `requirements.txt` — Python dependencies.

Quick start (Windows PowerShell):

```powershell
# create venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run the app
$env:FLASK_APP = 'flask_app.py'; flask run
```

Notes:
- The code assumes `lightgbm_job_scam_model.pkl` is in the project root (next to `flask_app.py`).
- The app also looks for `clean_fake_job_postings_structured.csv` to show a small dataset preview.
- The prediction endpoint expects the model object to accept a DataFrame with the fields listed in the form. If your model expects preprocessed features (vectorized text, etc.), save it as a Pipeline that contains preprocessing + estimator so this Flask app can call `predict`/`predict_proba` directly.
