import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"  # change this for production

# Supabase client setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# ROUTES
# ----------------------------

# Landing page
@app.route("/")
def index():
    return render_template("index.html")

# Login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        response = supabase.table("users").select("*").eq("email", email).execute()
        if response.data:
            user = response.data[0]
            stored_password = user["password"]

            if password == stored_password:  # ⚠️ plain text for now
                session["user"] = {"email": user["email"], "role": user["role"]}
                flash("Login successful!", "success")

                if user["role"] == "system_admin":
                    return redirect(url_for("admin_dashboard"))
                elif user["role"] == "doctor":
                    return redirect(url_for("doctor_dashboard"))
            else:
                flash("Invalid password", "danger")
        else:
            flash("User not found", "danger")

    return render_template("login.html")

# ----------------------------
# Admin routes
# ----------------------------

# Admin dashboard (single-page)
@app.route("/admin")
def admin_dashboard():
    if "user" in session and session["user"]["role"] == "system_admin":
        # Fetch full user record from Supabase using email
        response = supabase.table("users").select("*").eq("email", session["user"]["email"]).execute()
        if response.data:
            current_user = response.data[0]  # contains name, email, role
        else:
            current_user = session["user"]  # fallback

        # Fetch all users for the table
        users_response = supabase.table("users").select("*").execute()
        users = users_response.data if users_response.data else []

        return render_template("admin_dashboard.html", user=current_user, users=users)
    return redirect(url_for("login"))

# Add user (Admin only)
@app.route("/add_user", methods=["POST"])
def add_user():
    if "user" not in session or session["user"]["role"] != "system_admin":
        flash("Unauthorized access!", "danger")
        return redirect(url_for("login"))

    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    role = request.form.get("role")

    if not (name and email and password and role):
        flash("All fields are required!", "danger")
        return redirect(url_for("admin_dashboard"))

    try:
        supabase.table("users").insert({
            "name": name,
            "email": email,
            "password": password,
            "role": role
        }).execute()
        flash("User added successfully!", "success")
    except Exception as e:
        flash(f"Error adding user: {str(e)}", "danger")

    return redirect(url_for("admin_dashboard"))

# Delete user
@app.route("/delete_user/<user_id>", methods=["POST"])
def delete_user(user_id):
    if "user" not in session or session["user"]["role"] != "system_admin":
        return redirect(url_for("login"))

    try:
        supabase.table("users").delete().eq("id", user_id).execute()
        flash("User deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting user: {e}", "danger")

    return redirect(url_for("admin_dashboard"))

# ----------------------------
# Doctor routes
# ----------------------------
@app.route("/doctor")
def doctor_dashboard():
    if "user" in session and session["user"]["role"] == "doctor":
        # Fetch the full user record from Supabase
        response = supabase.table("users").select("*").eq("email", session["user"]["email"]).execute()
        if response.data:
            user = response.data[0]  # full user object including name
            return render_template("doc_dashboard.html", user=user)
    return redirect(url_for("login"))

@app.route("/diagnosis")
def diagnosis():
    if "user" in session and session["user"]["role"] == "doctor":
        return render_template("diagnosis.html")
    return redirect(url_for("login"))

@app.route("/history")
def history():
    if "user" in session and session["user"]["role"] == "doctor":
        return render_template("history.html")
    return redirect(url_for("login"))

@app.route("/patients")
def patients():
    if "user" in session and session["user"]["role"] == "doctor":
        return render_template("patients.html")
    return redirect(url_for("login"))



# ----------------------------
# Logout
# ----------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
