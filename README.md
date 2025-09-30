# PneumoAI: Pediatric X-ray Diagnostic Tool

## 🌟 Overview

**PneumoAI** is an AI-powered clinical decision support system designed to rapidly analyze pediatric chest X-ray images for the diagnosis of **Pneumonia** (Normal, Bacterial, or Viral). Built as a Flask web application, it provides a secure, intuitive interface for medical professionals (Doctors) to upload images, receive instantaneous AI predictions (including confidence scores and Grad-CAM visualization), and officially finalize case diagnoses.

This project focuses on delivering **high-confidence predictions** and **model interpretability**, ensuring the tool acts as a reliable assistant in fast-paced clinical environments.

---

## ✨ Features

* **Secure Authentication:** Role-based access control for Doctors (full features) and Technicians (upload only).
* **AI Diagnostics:** Instant classification of X-rays into **Normal**, **Bacterial Pneumonia**, or **Viral Pneumonia**.
* **Confidence Scoring:** Provides detailed percentage breakdowns for each class, highlighting cases that require mandatory physician review.
* **Model Interpretability (Grad-CAM):** Generates and displays heatmaps showing the specific regions of the X-ray the AI model focused on when making its prediction.
* **Professional Reporting:** One-click generation of a polished, print-ready report containing all patient, clinical, AI, and image data.
* **Case Management:** Doctors can save the AI prediction, add their official diagnosis and notes, and finalize the case record.

---

## 🛠️ Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | **Python 3** (Flask) | Lightweight web server and application logic. |
| **Database** | **Supabase** (PostgreSQL) | Scalable backend service for data storage (diagnoses, patients) and file storage (X-rays). |
| **AI/ML** | **TensorFlow / Keras** | Used for training and serving the CNN diagnostic model. |
| **PDF Generation** | **WeasyPrint** | Converts the prediction web page into a high-fidelity, professional PDF report. |
| **Front-end** | **HTML5, CSS3** (Jinja2 Templates) | Clean, intuitive, and professional UI/UX for clinical use. |
| **Image Processing** | **PIL/Pillow, Base64** | Handling image uploads and displaying Grad-CAM visualizations. |

---

