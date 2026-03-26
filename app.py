from __future__ import annotations

import os

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from config import Config
from extensions import bcrypt, csrf, db, login_manager
from forms import ClassifyEmailForm, LoginForm, RegisterForm
from ml.predict import Predictor
from models import ClassificationRecord, User


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)

    os.makedirs(app.instance_path, exist_ok=True)
    # Use an absolute SQLite path so the app can always open the DB file.
    if not os.environ.get("DATABASE_URL"):
        db_path = os.path.join(app.instance_path, "database.db")
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"

    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "login"
    login_manager.login_message_category = "warning"

    predictor = Predictor(model_path=os.environ.get("MODEL_PATH", "ml/model.json"))

    with app.app_context():
        db.create_all()

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.query.get(int(user_id))

    @app.get("/")
    def home():
        return render_template("home.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        form = RegisterForm()
        if form.validate_on_submit():
            email = form.email.data.strip().lower()
            existing = User.query.filter_by(email=email).first()
            if existing:
                flash("An account with that email already exists.", "danger")
                return render_template("register.html", form=form)
            pw_hash = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
            user = User(email=email, password_hash=pw_hash)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for("dashboard"))
        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        form = LoginForm()
        if form.validate_on_submit():
            email = form.email.data.strip().lower()
            user = User.query.filter_by(email=email).first()
            if not user or not bcrypt.check_password_hash(
                user.password_hash, form.password.data
            ):
                flash("Invalid email or password.", "danger")
                return render_template("login.html", form=form)
            login_user(user)
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard"))
        return render_template("login.html", form=form)

    @app.post("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("home"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        form = ClassifyEmailForm()

        total = ClassificationRecord.query.filter_by(user_id=current_user.id).count()
        spam = ClassificationRecord.query.filter_by(
            user_id=current_user.id, verdict="SPAM"
        ).count()
        ham = ClassificationRecord.query.filter_by(
            user_id=current_user.id, verdict="HAM"
        ).count()

        model_ready = True
        model_error = None
        try:
            if not predictor.ready:
                predictor.load()
        except Exception as e:
            model_ready = False
            model_error = str(e)

        return render_template(
            "dashboard.html",
            form=form,
            total=total,
            spam=spam,
            ham=ham,
            model_ready=model_ready,
            model_error=model_error,
        )

    @app.post("/classify")
    @login_required
    def classify():
        form = ClassifyEmailForm()
        if not form.validate_on_submit():
            flash("Please paste an email body (at least a few words).", "danger")
            return redirect(url_for("dashboard"))

        subject = (form.subject.data or "").strip()
        body = (form.body.data or "").strip()

        # If the ML model isn't trained/deployed yet, don't throw a 500.
        try:
            if not predictor.ready:
                predictor.load()
        except FileNotFoundError:
            flash(
                "ML model is not available yet. Please train/deploy `ml/model.json` first.",
                "warning",
            )
            return redirect(url_for("dashboard"))

        pred = predictor.predict(subject=subject, body=body)
        record = ClassificationRecord(
            user_id=current_user.id,
            subject=(subject[:60] if subject else None),
            body=body,
            full_text=f"{subject} {body}".strip(),
            verdict=pred.verdict,
            confidence=pred.confidence,
        )
        db.session.add(record)
        db.session.commit()
        return redirect(url_for("result", record_id=record.id))

    @app.get("/result/<int:record_id>")
    @login_required
    def result(record_id: int):
        record = ClassificationRecord.query.filter_by(
            id=record_id, user_id=current_user.id
        ).first_or_404()
        return render_template("result.html", record=record)

    @app.get("/history")
    @login_required
    def history():
        records = (
            ClassificationRecord.query.filter_by(user_id=current_user.id)
            .order_by(ClassificationRecord.created_at.desc())
            .all()
        )
        return render_template("history.html", records=records)

    @app.errorhandler(404)
    def not_found(_):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(_):
        return render_template("errors/500.html"), 500

    return app


app = create_app()


if __name__ == "__main__":
    # Local development entrypoint (do not use in production).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)

