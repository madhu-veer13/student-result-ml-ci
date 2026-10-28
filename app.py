import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from werkzeug.exceptions import HTTPException

# ---------------------------------------------------------------
# Logging configuration: write to stdout so Docker/Render capture it
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

app = Flask(__name__)

# ---------------------------------------------------------------
# Practical 12: Prometheus metrics (define each metric once, here)
# ---------------------------------------------------------------
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"],
)
MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded",
)
# Make the series exist (value 0) before the first request
PREDICTIONS.labels(result="success")
PREDICTIONS.labels(result="error")
MODEL_LOADED.set(0)

MODEL_PATH = Path("student_result_model.pkl")
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Only add non-sensitive features here if input values must be logged.
SAFE_INPUT_FIELDS = set()


def load_model():
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        logger.error("event=model_load_failed model_file=%s reason=not_found",
                     MODEL_PATH)
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)
    return model


# Log model loading at startup (does not stop the app if it fails)
try:
    load_model()
    logger.info("event=model_loaded model_file=%s", MODEL_PATH)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)


# ---------------------------------------------------------------
# Request / response logging
# ---------------------------------------------------------------
@app.before_request
def start_request():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    duration_ms = (time.perf_counter() - g.get("start_time", time.perf_counter())) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.get("request_id", "-"),
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = g.get("request_id", "-")

    # Practical 12: record metrics (skip Prometheus' own scrapes)
    if request.path != "/metrics":
        # Use the route pattern; unknown URLs share one label (avoids label explosion)
        path = request.url_rule.rule if request.url_rule else "unmatched"
        duration = time.perf_counter() - g.get("start_time", time.perf_counter())
        HTTP_REQUESTS.labels(
            method=request.method, path=path, status=str(response.status_code)
        ).inc()
        HTTP_LATENCY.labels(method=request.method, path=path).observe(duration)
    return response


@app.errorhandler(Exception)
def handle_exception(error):
    # Normal HTTP errors (404, 405, ...) keep their usual response;
    # they are logged as WARNING by log_request above.
    if isinstance(error, HTTPException):
        return error
    logger.exception("event=unhandled_exception request_id=%s path=%s",
                     g.get("request_id", "-"), request.path)
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/health")
@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("event=validation_error path=/predict reason=missing_json_body")
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error path=/predict missing_fields=%s error_count=%s",
            missing_fields,
            len(missing_fields),
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        safe_input = {
            key: data[key] for key in FEATURES if key in SAFE_INPUT_FIELDS
        }
        logger.info(
            "event=prediction_request request_id=%s feature_names=%s safe_input=%s",
            g.get("request_id", "-"),
            FEATURES,
            safe_input,
        )

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        PREDICTIONS.labels(result="success").inc()
        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.get("request_id", "-"),
            prediction,
        )
    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed request_id=%s",
                         g.get("request_id", "-"))
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
