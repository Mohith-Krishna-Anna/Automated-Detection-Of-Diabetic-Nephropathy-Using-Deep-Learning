from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import os
import joblib
from functools import wraps
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "dn_secret_key_2024"

# Simple in-memory user store: {username: password}
users = {}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "ann_model.h5")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Home — public landing page ───────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


# ── Register ──────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if "username" in session:
        return redirect(url_for("predict"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if not username or not password:
            flash("Username and password are required.", "error")
        elif username in users:
            flash("Username already taken. Choose another.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        else:
            users[username] = password
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("register.html")


# ── Login ─────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("predict"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username in users and users[username] == password:
            session["username"] = username
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("predict"))
        else:
            flash("Invalid username or password.", "error")
    return render_template("login.html")


# ── Logout ────────────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    username = session.pop("username", None)
    flash(f"You have been logged out{', ' + username if username else ''}.", "info")
    return redirect(url_for("login"))


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET"])
@login_required
def predict():
    return render_template("predict.html", username=session["username"])


@app.route("/predict", methods=["POST"])
@login_required
def predict_result():
    try:
        input_features = [float(x) for x in request.form.values()]
        input_array  = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction   = model.predict(input_scaled)[0][0]

        if prediction > 0.5:
            result     = "CKD Detected"
            result_type = "danger"
        else:
            result     = "No CKD Detected"
            result_type = "safe"

        confidence = round(float(prediction) * 100, 2)
        return render_template(
            "result.html",
            prediction_text=result,
            result_type=result_type,
            confidence=confidence,
            username=session["username"],
        )
    except Exception as e:
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for("predict"))


if __name__ == "__main__":
    app.run(debug=True)