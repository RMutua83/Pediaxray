import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from supabase import create_client, Client
from dotenv import load_dotenv
from io import BytesIO

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

@app.route("/")
def index():
    return render_template("index.html")

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

@app.route("/admin")
def admin_dashboard():
    if "user" in session and session["user"]["role"] == "system_admin":
        users_response = supabase.table("users").select("*").execute()
        users = users_response.data if users_response.data else []
        return render_template("admin_dashboard.html", users=users)
    return redirect(url_for("login"))

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
        return render_template("doc_dashboard.html")
    return redirect(url_for("login"))

@app.route("/patients")
def patients():
    if "user" in session and session["user"]["role"] == "doctor":
        doctor_email = session["user"]["email"]
        doctor_res = supabase.table("users").select("id").eq("email", doctor_email).execute()
        doctor_id = doctor_res.data[0]["id"] if doctor_res.data else None

        patients_res = supabase.table("patients").select("*").eq("doctor_id", doctor_id).execute()
        patients = patients_res.data if patients_res.data else []

        return render_template("patients.html", patients=patients)
    return redirect(url_for("login"))

@app.route("/patients/add", methods=["GET", "POST"])
def add_patient():
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    if request.method == "POST":
        patient_name = request.form.get("patient_name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        weight = request.form.get("weight")
        height = request.form.get("height")
        medical_history = request.form.get("medical_history")

        doctor_res = supabase.table("users").select("id").eq("email", session["user"]["email"]).execute()
        doctor_id = doctor_res.data[0]["id"] if doctor_res.data else None

        try:
            supabase.table("patients").insert({
                "name": patient_name,
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "medical_history": medical_history,
                "doctor_id": doctor_id
            }).execute()
            flash(f"Patient {patient_name} added successfully!", "success")
        except Exception as e:
            flash(f"Error adding patient: {str(e)}", "danger")

        return redirect(url_for("patients"))

    return render_template("add_patient.html")

@app.route("/diagnosis", methods=["GET", "POST"])
def diagnosis():
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    # For GET: list patients
    doctor_res = supabase.table("users").select("id").eq("email", session["user"]["email"]).execute()
    doctor_id = doctor_res.data[0]["id"] if doctor_res.data else None
    patients_res = supabase.table("patients").select("*").eq("doctor_id", doctor_id).execute()
    patients = patients_res.data if patients_res.data else []

    if request.method == "POST":
        # Form data
        patient_id = request.form.get("patient_id")
        patient_name = request.form.get("patient_name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        weight = request.form.get("weight")
        symptoms = request.form.get("symptoms")
        medical_history = request.form.get("medical_history")

        # Create new patient if none selected
        if not patient_id:
            patient_insert = supabase.table("patients").insert({
                "name": patient_name,
                "age": int(age) if age else None,
                "gender": gender,
                "weight": float(weight) if weight else None,
                "medical_history": medical_history,
                "doctor_id": doctor_id
            }).select("*").execute()  # <-- add .select("*") to get returned patient

            if patient_insert.data and len(patient_insert.data) > 0:
                patient_id = patient_insert.data[0]["id"]
            else:
                flash("Failed to create patient.", "danger")
                return redirect(url_for("diagnosis"))

        # Handle X-ray file
        xray_file = request.files.get("xray_image")
        xray_url = None
        if xray_file:
            ext = xray_file.filename.split(".")[-1]
            filename = f"{uuid.uuid4()}.{ext}"
            file_bytes = xray_file.read()
            try:
                supabase.storage.from_("xray-images").upload(filename, file_bytes)
                xray_url = supabase.storage.from_("xray-images").get_public_url(filename).public_url
            except Exception as e:
                flash(f"Error uploading X-ray: {str(e)}", "danger")
                return redirect(request.url)

        # Insert diagnosis
        insert_res = supabase.table("diagnoses").insert({
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "xray_url": xray_url,
            "symptoms": symptoms
        }).select("*").execute()  # <-- add .select("*") to get returned diagnosis

        if insert_res.data and len(insert_res.data) > 0:
            diagnosis_id = insert_res.data[0]["id"]
            return redirect(url_for("prediction", diagnosis_id=diagnosis_id))
        else:
            flash("Failed to create diagnosis record.", "danger")
            return redirect(url_for("diagnosis"))

    return render_template("diagnosis.html", patients=patients)

@app.route("/prediction/<diagnosis_id>")
def prediction(diagnosis_id):
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    diag_res = supabase.table("diagnoses").select("*").eq("id", diagnosis_id).execute()
    if not diag_res.data or len(diag_res.data) == 0:
        flash("Diagnosis not found.", "danger")
        return redirect(url_for("diagnosis"))

    diagnosis = diag_res.data[0]

    patient_res = supabase.table("patients").select("*").eq("id", diagnosis["patient_id"]).execute()
    patient = patient_res.data[0] if patient_res.data else {}

    gradcam_url = "/static/sample_gradcam.png"  # Placeholder
    confidence = 95.0  # Placeholder

    diagnosis_obj = {
        "id": diagnosis.get("id"),
        "patient_id": patient.get("id"),
        "patient_name": patient.get("name"),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "weight": patient.get("weight"),
        "medical_history": patient.get("medical_history"),
        "symptoms": diagnosis.get("symptoms"),
        "xray_url": diagnosis.get("xray_url")
    }

    return render_template("prediction.html",
                           diagnosis=diagnosis_obj,
                           gradcam_url=gradcam_url,
                           confidence=confidence)

# History and download routes remain unchanged

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
