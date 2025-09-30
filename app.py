import os
import uuid
import csv
import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
import cv2 # OpenCV for Grad-CAM visualization
from PIL import Image
from io import StringIO, BytesIO

# Flask essentials and extensions
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, send_file
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Database & Flask Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase environment variables (SUPABASE_URL, SUPABASE_KEY) are not set.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey_dev_default")

# --- Global System Settings ---
GLOBAL_MIN_CONFIDENCE_THRESHOLD = 75.0 # For flagging results for review
GLOBAL_APP_STATUS = 'ONLINE'
PREDICTION_COLUMN = "prediction_result" 
CONFIDENCE_COLUMN = "prediction_confidence"

# --- Machine Learning Model Setup ---
MODEL_PATH = 'custom_cnn_pneumo_ai_best_baseline.keras'
IMG_SIZE = (128, 128)
CLASS_LABELS = ['NORMAL', 'BACTERIAL', 'VIRAL'] 
# <<< CRITICAL: REPLACE THIS VALUE WITH YOUR MODEL'S ACTUAL LAST CONV LAYER NAME! >>>
LAST_CONV_LAYER_NAME = 'conv2d_15' # 'conv2d_3'

try:
    pneumo_model = load_model(MODEL_PATH, compile=False) 
    print(f"Pneumonia Model loaded successfully from: {MODEL_PATH}")
    
except Exception as e:
    pneumo_model = None
    print(f"ERROR: Could not load the model from {MODEL_PATH}. Prediction disabled: {e}")

# --- Utility Function for Grad-CAM ---

def get_grad_cam_heatmap(img_array, model, last_conv_layer_name, predicted_index):
    """Generates the Grad-CAM heatmap for a specific predicted class index."""
    if model is None:
        return None

    # 1. Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # 2. Compute the gradient of the top predicted class
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        pred_index = predicted_index 
        class_channel = preds[:, pred_index]

    # 3. Gradient of the output neuron
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Vector of mean intensity (weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel's activation by its weight
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()

# ---------------------------------------------------------------------


# ----------------------------
# BASE & LOGIN ROUTES
# ----------------------------

@app.route("/")
def index():
    if "user" in session:
        if session["user"]["role"] == "system_admin":
            return redirect(url_for("admin_dashboard"))
        elif session["user"]["role"] == "doctor":
            return redirect(url_for("doc_dashboard"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        try:
            res = supabase.table("users").select("*").eq("email", email).limit(1).execute()
            user_data = res.data[0] if res.data else None
            
            # WARNING: Insecure password check - replace with hashed password check in production!
            if user_data and user_data['password'] == password: 
                # Store full user data including ID for foreign key inserts
                session["user"] = {
                    "id": user_data['id'],
                    "name": user_data.get('name', user_data['email']),
                    "email": user_data['email'],
                    "role": user_data['role']
                }
                flash(f"Logged in successfully as {user_data['role']}.", "success")
                if user_data['role'] == "system_admin":
                    return redirect(url_for("admin_dashboard"))
                else:
                    return redirect(url_for("doc_dashboard"))
            else:
                flash("Invalid email or password.", "danger")
        except Exception as e:
            flash("Login error. Please try again.", "danger")
            print(f"LOGIN ERROR: {e}")
            
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ----------------------------
# ADMIN ROUTES
# ----------------------------
@app.route("/admin")
def admin_dashboard():
    if "user" not in session or session["user"]["role"] != "system_admin":
        return redirect(url_for("login"))
    
    current_user = session["user"]
    
    try:
        # 1. Fetch ALL Users 
        users_response = supabase.table("users").select("id, name, email, role").execute()
        users = users_response.data if users_response.data else []
        user_name_lookup = {u['id']: u['name'] for u in users}

        # 2. Fetch ALL Diagnoses for Stat Calculation
        all_diagnoses_res = supabase.table("diagnoses").select(f"{PREDICTION_COLUMN}, patient_id").execute()
        all_diagnoses = all_diagnoses_res.data if all_diagnoses_res.data else []

        # 3. Calculate Summary Stats (KPIs)
        total_users = len(users)
        total_diagnoses = len(all_diagnoses)
        
        # Count BACTERIAL and VIRAL as 'Pneumonia'
        pneumonia_count = sum(
            1 for d in all_diagnoses 
            if (d.get(PREDICTION_COLUMN) or '').upper() in ('BACTERIAL', 'VIRAL')
        )
        
        pending_count = sum(
            1 for d in all_diagnoses 
            if (d.get(PREDICTION_COLUMN) or '').upper() == 'PENDING'
        )
        
        # 4. Fetch Recent Activity 
        activity_res = supabase.table("diagnoses") \
            .select(f"*, patient_id, doctor_id, {PREDICTION_COLUMN}, {CONFIDENCE_COLUMN}") \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()

        recent_activity = activity_res.data if activity_res.data else []
        
        # 5. Fetch Patient Names 
        patient_ids = [d['patient_id'] for d in all_diagnoses if d.get('patient_id')]
        unique_patient_ids = list(set(patient_ids))
        
        patient_res = supabase.table("patients").select("id, name").in_("id", unique_patient_ids).execute()
        patient_name_lookup = {p['id']: p['name'] for p in patient_res.data}
        
        # 6. Process data for the template
        for item in recent_activity:
            if item.get("created_at"):
                 parts = item["created_at"].split("T")
                 item["display_time"] = f"{parts[0]} {parts[1].split('.')[0]}" 
            else:
                item["display_time"] = "N/A"
            
            p_id = item.get('patient_id')
            item['patients'] = {'name': patient_name_lookup.get(p_id, 'N/A')}
            
            d_id = item.get('doctor_id')
            item['users'] = {'name': user_name_lookup.get(d_id, 'N/A')}

            item['prediction'] = item.get(PREDICTION_COLUMN, 'N/A')
            item['confidence'] = item.get(CONFIDENCE_COLUMN, 0)
            

        # 7. Render the template
        return render_template("admin_dashboard.html", 
                               user=current_user,
                               users=users,
                               recent_activity=recent_activity,
                               total_users=total_users,
                               total_diagnoses=total_diagnoses,
                               pneumonia_count=pneumonia_count,
                               pending_count=pending_count,
                               global_min_confidence_threshold=GLOBAL_MIN_CONFIDENCE_THRESHOLD,
                               app_status=GLOBAL_APP_STATUS)

    except Exception as e:
        print(f"ADMIN DASHBOARD CRITICAL FAILURE: {e}") 
        flash(f"A critical error occurred while fetching data: {str(e)}", "danger")
        
        return render_template("admin_dashboard.html", 
                               user=current_user, 
                               users=[], 
                               recent_activity=[],
                               total_users=0, total_diagnoses=0, pneumonia_count=0, pending_count=0,
                               global_min_confidence_threshold=GLOBAL_MIN_CONFIDENCE_THRESHOLD,
                               app_status=GLOBAL_APP_STATUS
                               )


@app.route("/admin/toggle_system_status", methods=["POST"])
def toggle_system_status():
    if "user" not in session or session["user"]["role"] != "system_admin":
        flash("Unauthorized access.", "danger")
        return redirect(url_for("login"))

    global GLOBAL_APP_STATUS
    
    if GLOBAL_APP_STATUS == 'ONLINE':
        GLOBAL_APP_STATUS = 'MAINTENANCE'
        flash("Application is now in **Maintenance Mode**. New diagnosis submissions are blocked.", "warning")
    else:
        GLOBAL_APP_STATUS = 'ONLINE'
        flash("Application is **Online** and fully operational.", "success")
        
    return redirect(url_for("admin_dashboard") + "#settings-section")


@app.route("/admin/save_core_settings", methods=["POST"])
def save_core_settings():
    if "user" not in session or session["user"]["role"] != "system_admin":
        flash("Unauthorized access.", "danger")
        return redirect(url_for("login"))

    global GLOBAL_MIN_CONFIDENCE_THRESHOLD
    
    try:
        new_conf = request.form.get("min_conf")
        
        if new_conf:
            new_conf = float(new_conf)
            if 50 <= new_conf <= 100:
                GLOBAL_MIN_CONFIDENCE_THRESHOLD = new_conf
                flash(f"AI Confidence Threshold updated to {new_conf}%.", "success")
            else:
                flash("Confidence threshold must be between 50 and 100.", "warning")
        else:
             flash("Confidence value missing.", "warning")

    except ValueError:
        flash("Invalid value provided for confidence threshold.", "danger")
    
    return redirect(url_for("admin_dashboard") + "#settings-section")


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
# DOCTOR ROUTES
# ----------------------------

@app.route("/doctor")
def doctor_dashboard():
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))
    
    current_user = session["user"]
    patients = [] 
    recent_diagnoses = []
    
    try:
        # Fetch patients linked to this doctor
        patients_res = supabase.table("patients").select("*").eq("doctor_id", current_user["id"]).order("name", desc=False).execute()
        patients = patients_res.data if patients_res.data else []
        
        # Fetch recent diagnoses by this doctor (10 most recent)
        diagnoses_res = supabase.table("diagnoses") \
            .select(f"*, patients(name), {PREDICTION_COLUMN}, {CONFIDENCE_COLUMN}") \
            .eq("doctor_id", current_user["id"]) \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        
        recent_diagnoses = diagnoses_res.data if diagnoses_res.data else []
        
    except Exception as e:
        print(f"DOCTOR DASHBOARD ERROR: {e}")
        flash("Error fetching data.", "danger")
        
    return render_template("doc_dashboard.html", # Renamed to doctor_dashboard.html for consistency
                           user=current_user, 
                           patients=patients, 
                           recent_diagnoses=recent_diagnoses,
                           global_min_confidence_threshold=GLOBAL_MIN_CONFIDENCE_THRESHOLD,
                           app_status=GLOBAL_APP_STATUS)


@app.route("/patients")
def patients():
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    doctor_id = session["user"]["id"]
    try:
        patients_res = supabase.table("patients").select("*").eq("doctor_id", doctor_id).execute()
        patients = patients_res.data if patients_res.data else []

        return render_template("patients.html", patients=patients)
    except Exception as e:
        flash(f"Database error fetching patients: {str(e)}", "danger")
        return redirect(url_for("doc_dashboard"))


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
        doctor_id = session["user"]["id"]
        
        try:
            supabase.table("patients").insert({
                "name": patient_name,
                "age": int(age) if age else None,
                "gender": gender,
                "weight": float(weight) if weight else None,
                "height": float(height) if height else None,
                "medical_history": medical_history,
                "doctor_id": doctor_id
            }).execute()
            flash(f"Patient {patient_name} added successfully!", "success")
        except Exception as e:
            flash(f"Error adding patient: {str(e)}", "danger")

        return redirect(url_for("patients"))

    return render_template("add_patient.html")


# =========================================================================
# DOCTOR ROUTE: /diagnosis (Handles GET for form, POST for submission)
# =========================================================================

@app.route("/diagnosis", methods=["GET", "POST"])
def diagnosis():
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))
    
    doctor_id = session["user"]["id"]
    
    # 1. Fetch patient list for the form (GET and POST context)
    try:
        patients_res = supabase.table("patients").select("*").eq("doctor_id", doctor_id).execute()
        patients = patients_res.data if patients_res.data else []
    except Exception as e:
        flash(f"Database error fetching patient list: {str(e)}", "danger")
        patients = []

    if request.method == "POST":
        # Check system status
        if pneumo_model is None:
            flash("AI Model is not loaded. Cannot run diagnosis.", "danger")
            return redirect(url_for("diagnosis"))
        if GLOBAL_APP_STATUS != 'ONLINE':
            flash("System is in maintenance. Diagnosis submission is blocked.", "warning")
            return redirect(url_for("diagnosis"))

        patient_id = request.form.get("patient_id")
        symptoms = request.form.get("symptoms")
        xray_file = request.files.get("xray_image")

        if not patient_id:
            flash("Please select an existing patient ID.", "danger")
            return redirect(url_for("diagnosis"))

        if not (xray_file and xray_file.filename):
            flash("X-ray image is required for analysis.", "danger")
            return redirect(url_for("diagnosis"))

        file_bytes = None
        xray_url = None
        
        # --- File Upload & Storage ---
        try:
            # Read file bytes once for use in both storage and AI prediction
            file_bytes = xray_file.read()
            if not file_bytes:
                flash("Error reading X-ray file.", "danger")
                return redirect(url_for("diagnosis"))

            ext = xray_file.filename.split(".")[-1]
            filename = f"xray_{patient_id}_{uuid.uuid4()}.{ext}"
            
            # 1. Upload the file to Supabase Storage
            # Use the 'file_bytes' content read above
            supabase.storage.from_("xray-images").upload(filename, file_bytes)
            
            # 2. Get the public URL
            xray_url = supabase.storage.from_("xray-images").get_public_url(filename)
            
        except Exception as e:
            flash(f"Critical error during X-ray upload/URL retrieval: {str(e)}", "danger")
            print(f"X-RAY UPLOAD/URL FAILED: {e}")
            return redirect(url_for("diagnosis"))


        # --- AI Prediction Logic ---
        img_array = None
        try:
            # 3. Preprocess the image
            img = Image.open(BytesIO(file_bytes)).convert('RGB')
            # Use the correct size (128, 128) as per your training code
            img = img.resize(IMG_SIZE) 
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0 # Normalize

        except Exception as e:
            flash(f"Error processing image for AI model: {e}", "danger")
            return redirect(url_for("diagnosis"))

        # 4. Perform Prediction
        predictions = pneumo_model.predict(img_array)[0] 
        predicted_index = np.argmax(predictions) 
        prediction_text = CLASS_LABELS[predicted_index] 
        confidence = float(predictions[predicted_index] * 100)
        
        # Generate the full confidence breakdown dictionary (for JSONB column)
        confidence_breakdown = {}
        for i, label in enumerate(CLASS_LABELS):
            confidence_breakdown[label] = round(float(predictions[i] * 100), 2)
        

        # 5. Generate Grad-CAM for the Predicted Class (USING CORRECTED LAYER NAME)
        gradcam_base64 = ""
        try:
            # 🚨 LAST_CONV_LAYER_NAME MUST BE SET TO 'conv2d_15'
            heatmap = get_grad_cam_heatmap(img_array, pneumo_model, LAST_CONV_LAYER_NAME, predicted_index)
            
            # Prepare image for overlay
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Resize heatmap to match image size and overlay
            heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
            
            # Encode the overlaid image to Base64
            _, buffer = cv2.imencode('.png', superimposed_img)
            gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            print(f"GRAD-CAM ERROR: {e}")
            flash("Failed to generate Grad-CAM visualization.", "warning")


        # 6. Store Result in Supabase Database
        try:
            data_to_insert = {
                "doctor_id": doctor_id,
                "patient_id": patient_id,
                "xray_url": xray_url,
                "symptoms": symptoms,
                "prediction_result": prediction_text,
                "prediction_confidence": round(confidence, 2),
                "prediction_breakdown": confidence_breakdown,
                "gradcam_base64": gradcam_base64
            }
            
            # CORRECTED: Insert data and retrieve the inserted row
            insert_res = supabase.table("diagnoses").insert(data_to_insert).execute()
            
            if insert_res.data and len(insert_res.data) > 0:
                diagnosis_id = insert_res.data[0]['id']
                flash("Diagnosis processed and saved successfully.", "success")
                return redirect(url_for("prediction", diagnosis_id=diagnosis_id))
            else:
                raise Exception("Insert failed to return data.")
            
        except Exception as e:
            flash(f"Database insertion failed: {e}", "danger")
            print(f"DATABASE INSERTION ERROR: {e}")
            return redirect(url_for("diagnosis"))

    return render_template("diagnosis.html", patients=patients)

@app.route("/prediction/<diagnosis_id>")
def prediction(diagnosis_id):
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    try:
        # Fetch all required columns including prediction_breakdown
        diag_res = supabase.table("diagnoses") \
            .select(f"*, patients(*), {PREDICTION_COLUMN}, {CONFIDENCE_COLUMN}, prediction_breakdown, gradcam_base64") \
            .eq("id", str(diagnosis_id)) \
            .single() \
            .execute()
        
        diagnosis = diag_res.data
        if not diagnosis:
            flash("Diagnosis record not found.", "danger")
            return redirect(url_for("doc_dashboard"))
        
        # Prepare data for template
        needs_review = diagnosis.get(CONFIDENCE_COLUMN, 0) < GLOBAL_MIN_CONFIDENCE_THRESHOLD

        return render_template("prediction.html", 
                               diagnosis=diagnosis,
                               needs_review=needs_review,
                               class_labels=CLASS_LABELS,
                               global_min_confidence_threshold=GLOBAL_MIN_CONFIDENCE_THRESHOLD,
                               user=session["user"])

    except Exception as e:
        flash(f"Error retrieving prediction: {str(e)}", "danger")
        print(f"PREDICTION VIEW ERROR: {e}")
        return redirect(url_for("doctor_dashboard"))


@app.route("/history")
def history():
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    doctor_id = session["user"]["id"]
    try:
        # 1. Fetch diagnoses linked to the doctor, joining to get ALL patient details.
        history_res = supabase.table("diagnoses") \
            .select("*, patients(*)") \
            .eq("doctor_id", doctor_id) \
            .order("created_at", desc=True) \
            .execute()
        
        history_records = history_res.data if history_res.data else []

        # Process the timestamps
        for record in history_records:
            if record.get("created_at"):
                parts = record["created_at"].split("T")
                record["display_date"] = parts[0]
                record["display_time"] = parts[1].split(".")[0]
            else:
                record["display_date"] = "N/A"
                record["display_time"] = "N/A"

        return render_template("history.html", history_records=history_records)
        
    except Exception as e:
        flash(f"Database error fetching history: {str(e)}", "danger")
        print(f"HISTORY FETCH ERROR: {e}")
        return redirect(url_for("doc_dashboard"))

# ----------------------------------------------------
# EXPORT ROUTES (History and Single Prediction Report)
# ----------------------------------------------------

@app.route("/export_history")
def export_history():
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    doctor_id = session["user"]["id"]
    try:
        # Fetch ALL diagnosis data 
        history_res = supabase.table("diagnoses") \
            .select(f"*, patients(*), {PREDICTION_COLUMN}, {CONFIDENCE_COLUMN}, prediction_breakdown") \
            .eq("doctor_id", doctor_id) \
            .order("created_at", desc=True) \
            .execute()
        
        history_records = history_res.data if history_res.data else []
        
        # Start CSV generation
        si = StringIO()
        cw = csv.writer(si)

        # Define CSV Header including the new detailed breakdown
        header = [
            'Diagnosis ID', 'Date', 'Time', 'Patient Name', 'Patient Age', 'Symptoms', 
            'AI Prediction', 'Confidence (%)', 'Conf_NORMAL', 'Conf_BACTERIAL', 'Conf_VIRAL', 
            'X-ray URL'
        ]
        cw.writerow(header)

        # Write Data Rows
        for record in history_records:
            patient = record.get('patients', {})
            breakdown = record.get('prediction_breakdown', {})
            
            timestamp_parts = record.get('created_at', 'N/A T N/A').split('T')
            date_part = timestamp_parts[0]
            time_part = timestamp_parts[1].split('.')[0] if len(timestamp_parts) > 1 else 'N/A'

            row = [
                record.get('id', ''),
                date_part,
                time_part,
                patient.get('name', 'Unknown'),
                patient.get('age', 'N/A'),
                record.get('symptoms', ''),
                record.get(PREDICTION_COLUMN, 'Pending'),
                record.get(CONFIDENCE_COLUMN, 'N/A'),
                breakdown.get('NORMAL', 'N/A'),
                breakdown.get('BACTERIAL', 'N/A'),
                breakdown.get('VIRAL', 'N/A'),
                record.get('xray_url', '')
            ]
            cw.writerow(row)

        output = si.getvalue()
        
        return Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment;filename=PneumoAI_Diagnosis_History.csv"
            }
        )

    except Exception as e:
        flash(f"Error exporting report: {str(e)}", "danger")
        print(f"EXPORT ERROR: {e}")
        return redirect(url_for("history"))


@app.route("/export_prediction_report/<diagnosis_id>")
def export_prediction_report(diagnosis_id):
    if "user" not in session or session["user"]["role"] != "doctor":
        return redirect(url_for("login"))

    try:
        # 1. Fetch the single diagnosis record and all patient details
        report_res = supabase.table("diagnoses") \
            .select(f"*, patients(*), {PREDICTION_COLUMN}, {CONFIDENCE_COLUMN}, prediction_breakdown") \
            .eq("id", str(diagnosis_id)) \
            .single() \
            .execute()
        
        record = report_res.data
        if not record:
            flash("Diagnosis record not found for report generation.", "danger")
            return redirect(url_for("doctor_dashboard"))

        patient = record.get('patients', {})
        breakdown = record.get('prediction_breakdown', {})

        # 2. Start CSV generation
        si = StringIO()
        cw = csv.writer(si)

        # Define CSV Header
        header = [
            'Diagnosis ID', 'Date', 'Patient Name', 'Age', 'Gender', 
            'Symptoms', 'Medical History', 'AI Prediction', 'Confidence (%)', 
            'Conf_NORMAL', 'Conf_BACTERIAL', 'Conf_VIRAL', 'X-ray URL'
        ]
        cw.writerow(header)

        # 3. Write Data Row
        timestamp_parts = record.get('created_at', 'N/A T N/A').split('T')
        date_part = timestamp_parts[0]
        
        row = [
            record.get('id', ''),
            date_part,
            patient.get('name', 'Unknown'),
            patient.get('age', 'N/A'),
            patient.get('gender', 'N/A'),
            record.get('symptoms', ''),
            patient.get('medical_history', ''),
            record.get(PREDICTION_COLUMN, 'Pending'),
            record.get(CONFIDENCE_COLUMN, 'N/A'),
            breakdown.get('NORMAL', 'N/A'),
            breakdown.get('BACTERIAL', 'N/A'),
            breakdown.get('VIRAL', 'N/A'),
            record.get('xray_url', '')
        ]
        cw.writerow(row)

        # 4. Create the Flask Response object
        output = si.getvalue()
        filename = f"PneumoAI_Report_{patient.get('name', 'Patient')}_{date_part}.csv"
        
        return Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename={filename}"
            }
        )

    except Exception as e:
        flash(f"Error exporting prediction report: {str(e)}", "danger")
        print(f"PREDICTION EXPORT ERROR: {e}")
        return redirect(url_for("prediction", diagnosis_id=diagnosis_id))


if __name__ == "__main__":
    # Note: On production systems, set debug=False
    app.run(debug=True)