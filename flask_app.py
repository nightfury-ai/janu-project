import os
import pickle
from pathlib import Path
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
try:
    # prefer joblib if the model was saved with joblib
    import joblib  # type: ignore
except Exception:
    joblib = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lightgbm_job_scam_model.pkl"
DATASET_PATH = BASE_DIR / "clean_fake_job_postings_structured.csv"

app = Flask(__name__)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    # Try joblib first (common for sklearn/lightgbm pipelines), then pickle
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


class DummyModel:
    """A tiny deterministic fallback model for local testing when the real model
    cannot be imported (for example if lightgbm isn't installed).

    - predict(X): returns 0 (not fraudulent) for every row
    - predict_proba(X): returns [0.9, 0.1] for every row (10% fraud probability)
    """
    def predict(self, X):
        # X can be a DataFrame or list-like; return zeros
        try:
            n = len(X)
        except Exception:
            n = 1
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        # return shape (n, 2)
        return np.vstack([[0.9, 0.1]] * n)


MODEL = None
try:
    MODEL = load_model(MODEL_PATH)
except Exception as e:
    # Keep app running; prediction endpoints will return a helpful error
    MODEL = None
    _model_load_error = str(e)
    # Optional: allow a quick dummy fallback to enable testing without LightGBM
    use_dummy = os.getenv("USE_DUMMY_MODEL", "0").lower() in ("1", "true", "yes")
    if use_dummy:
        MODEL = DummyModel()
        _model_load_error = f"Using dummy model because loading real model failed: {e}"


FEATURES = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "salary_min",
    "salary_max",
]


def coerce_df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any non-numeric pandas dtypes to integer codes so models that
    require numeric-only inputs (like LightGBM trained on numeric features)
    won't error during prediction.

    Strategy:
    - object / string columns -> pd.Categorical(...).codes (ints, -1 for NaN)
    - bool -> int
    - numeric -> keep as-is (cast to float)
    Returns a copy of df with numeric dtypes.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_bool_dtype(ser):
            out[col] = ser.astype(int)
        elif pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
            out[col] = pd.to_numeric(ser, errors="coerce").astype(float)
        else:
            # object, category, mixed types -> convert to categorical codes
            try:
                cat = pd.Categorical(ser.fillna("__MISSING__"))
                codes = cat.codes.astype(int)
            except Exception:
                # fallback: length of string
                codes = ser.fillna("").astype(str).str.len().astype(int)
            out[col] = codes
    return out


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    prob = None

    if request.method == "POST":
        if MODEL is None:
            error = f"Model not loaded: {_model_load_error}"
        else:
            # collect form data into a single-row DataFrame
            data = {}
            for f in FEATURES:
                val = request.form.get(f)
                # simple numeric conversion for salary fields and binary flags
                if f in ("salary_min", "salary_max"):
                    try:
                        data[f] = float(val) if val not in (None, "") else None
                    except Exception:
                        data[f] = None
                elif f in ("telecommuting", "has_company_logo", "has_questions"):
                    # allow user to submit '1' or '0' or leave blank
                    if val in ("1", "0"):
                        data[f] = int(val)
                    else:
                        data[f] = 1 if val and val.lower() in ("yes", "true", "y") else 0
                else:
                    data[f] = val if val is not None else ""

            df = pd.DataFrame([data], columns=FEATURES)

            try:
                # Many saved models expect numeric input dtypes. Convert object columns
                # to numeric codes so prediction won't fail with the pandas dtype error.
                df_numeric = coerce_df_to_numeric(df)

                # If model supports predict_proba, return probability for class 1
                if hasattr(MODEL, "predict_proba"):
                    proba = MODEL.predict_proba(df_numeric)
                    # pick probability of positive class (if binary)
                    if proba.shape[1] == 2:
                        prob = float(proba[0, 1])
                    else:
                        prob = None
                pred = MODEL.predict(df_numeric)
                # some models return array-like
                pred_label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
                result = {
                    "prediction": pred_label,
                    "label": ("Fraudulent" if pred_label == 1 else "Not Fraudulent"),
                    "probability": prob,
                }
            except Exception as e:
                error = f"Prediction error: {e}"

    # For GET and after POST render the form
    # Attempt to show a small CSV preview if available
    sample = None
    if DATASET_PATH.exists():
        try:
            sample = pd.read_csv(DATASET_PATH, nrows=5)
        except Exception:
            sample = None

    return render_template("index.html", features=FEATURES, result=result, error=error, sample=sample)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"error": f"Model not loaded: {_model_load_error}"}), 500

    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "JSON payload required"}), 400

    data = {}
    for f in FEATURES:
        val = payload.get(f)
        if f in ("salary_min", "salary_max"):
            try:
                data[f] = float(val) if val not in (None, "") else None
            except Exception:
                data[f] = None
        elif f in ("telecommuting", "has_company_logo", "has_questions"):
            if val in (1, "1", True, "true", "True"):
                data[f] = 1
            else:
                data[f] = 0
        else:
            data[f] = val if val is not None else ""

    df = pd.DataFrame([data], columns=FEATURES)
    try:
        prob = None
        if hasattr(MODEL, "predict_proba"):
            p = MODEL.predict_proba(df)
            if p.shape[1] == 2:
                prob = float(p[0, 1])
        pred = MODEL.predict(df)
        pred_label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
        return jsonify({"prediction": pred_label, "probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local debugging
    app.run(host="0.0.0.0", port=5000, debug=True)
