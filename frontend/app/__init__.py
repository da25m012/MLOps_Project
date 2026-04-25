"""Flask application factory."""

from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config["BACKEND_URL"] = "http://backend:8000"

    from .routes import bp
    app.register_blueprint(bp)

    return app
