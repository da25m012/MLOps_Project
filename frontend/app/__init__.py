"""Flask application factory."""

import os
from flask import Flask, Response
from .metrics import get_metrics, backend_health_status, track_request


def create_app():
    app = Flask(__name__)
    app.config["BACKEND_URL"] = os.environ.get("BACKEND_URL", "http://backend:8000")
    app.config["AIRFLOW_URL"] = os.environ.get("AIRFLOW_URL", "http://host.docker.internal:8080")

    from .routes import bp
    app.register_blueprint(bp)

    # Prometheus metrics endpoint
    @app.route("/metrics")
    def metrics():
        data, content_type = get_metrics()
        return Response(data, mimetype=content_type)

    # After each request update backend health gauge
    @app.after_request
    def after_request(response):
        return response

    return app
