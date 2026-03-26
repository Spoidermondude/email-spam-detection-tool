from __future__ import annotations

from datetime import datetime

from flask_login import UserMixin

from extensions import db


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    records = db.relationship(
        "ClassificationRecord", backref="user", lazy=True, cascade="all, delete-orphan"
    )


class ClassificationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    subject = db.Column(db.String(255), nullable=True)
    body = db.Column(db.Text, nullable=False)
    full_text = db.Column(db.Text, nullable=False)

    verdict = db.Column(db.String(10), nullable=False)  # "SPAM" | "HAM"
    confidence = db.Column(db.Float, nullable=False)  # 0..1

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

