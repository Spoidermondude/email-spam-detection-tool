from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length


class RegisterForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=8, max=72)]
    )
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=72)])
    submit = SubmitField("Log in")


class ClassifyEmailForm(FlaskForm):
    subject = StringField("Subject (optional)", validators=[Length(max=255)])
    body = TextAreaField("Email body", validators=[DataRequired(), Length(min=5)])
    submit = SubmitField("Analyze Email")

