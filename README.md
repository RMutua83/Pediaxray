# PneumoAI: Pediatric X-ray Diagnostic Tool
<img width="1881" height="1049" alt="image" src="https://github.com/user-attachments/assets/501da1ed-fb4c-4662-baf0-87708e8a308e" />
##  Overview

**PneumoAI** is an AI-powered clinical decision support system designed to rapidly analyze pediatric chest X-ray images for the diagnosis of **Pneumonia** (Normal, Bacterial, or Viral). Built as a Flask web application, it provides a secure, intuitive interface for medical professionals (Doctors) to upload images, receive instantaneous AI predictions (including confidence scores and Grad-CAM visualization), and officially finalize case diagnoses.

This project focuses on delivering **high-confidence predictions** and **model interpretability**, ensuring the tool acts as a reliable assistant in fast-paced clinical environments.

---

##  Features

* **Secure Authentication:** Role-based access control for Doctors (full features) and Technicians (upload only).
* **AI Diagnostics:** Instant classification of X-rays into **Normal**, **Bacterial Pneumonia**, or **Viral Pneumonia**.
* **Confidence Scoring:** Provides detailed percentage breakdowns for each class, highlighting cases that require mandatory physician review.
* **Model Interpretability (Grad-CAM):** Generates and displays heatmaps showing the specific regions of the X-ray the AI model focused on when making its prediction.
* **Data Export (CSV):** One-click generation of a **CSV file** containing structured case data, ideal for bulk analysis and record-keeping.
* **Case Management:** Doctors can save the AI prediction, add their official diagnosis and notes, and finalize the case record.

---

##  Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | **Python 3** (Flask) | Lightweight web server and application logic. |
| **Database** | **Supabase** (PostgreSQL) | Scalable backend service for data storage (diagnoses, patients) and file storage (X-rays). |
| **AI/ML** | **TensorFlow / Keras** | Used for training and serving the CNN diagnostic model. |
| **Data Processing** | **`csv`, `io.StringIO`** | Handles the efficient in-memory creation and streaming of the CSV data export. |
| **Front-end** | **HTML5, CSS3** (Jinja2 Templates) | Clean, intuitive, and professional UI/UX for clinical use. |
| **Image Processing** | **PIL/Pillow, Base64** | Handling image uploads and displaying Grad-CAM visualizations. |

---
## Screenshoots

<img width="1552" height="934" alt="image" src="https://github.com/user-attachments/assets/96b25557-1545-4872-a37d-1f2e79c10827" />
<img width="1888" height="1057" alt="image" src="https://github.com/user-attachments/assets/ec174bcf-8c2b-4d49-9e64-41934cd2fbea" />
<img width="1884" height="1054" alt="image" src="https://github.com/user-attachments/assets/a15ecca6-48d0-4b2d-a371-d197a66403af" />
<img width="1897" height="1024" alt="image" src="https://github.com/user-attachments/assets/51db85a0-bef1-4233-966d-647f426ca77c" />
<img width="1881" height="926" alt="image" src="https://github.com/user-attachments/assets/d5d726f4-507f-4809-9c02-2c0d83b4ffe5" />
<img width="1492" height="1062" alt="image" src="https://github.com/user-attachments/assets/f6871543-b518-4073-adde-5a81b37e077e" />
<img width="1052" height="843" alt="image" src="https://github.com/user-attachments/assets/53c48e99-a5c0-440c-a4b1-d513a7914d22" />
<img width="1897" height="1024" alt="image" src="https://github.com/user-attachments/assets/aeaa7e8e-7986-4979-9cc6-788b13868949" />











