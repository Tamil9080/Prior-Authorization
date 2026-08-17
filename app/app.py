import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import pypdf

# Add the parent folder to Python path to import correctly
script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from api.check import check_blueprint
from services.rag_service import rag_engine
from utils import audit_logger

app = Flask(__name__, static_folder="../static", static_url_path="")
app.register_blueprint(check_blueprint)

# Models paths in the refactored structure
CLINICAL_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest" / "clinical_rf_model.joblib"
REVIEW_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest" / "review_rf_model.joblib"

# Load models placeholders
clinical_model = None
review_model = None

# Initialize resources
CLINICAL_POLICY_TEXT = ""
MEMBER_HANDBOOK_TEXT = ""

def initialize_application():
    global clinical_model, review_model, CLINICAL_POLICY_TEXT, MEMBER_HANDBOOK_TEXT
    
    # 1. Load trained models
    if CLINICAL_MODEL_PATH.exists():
        try:
            clinical_model = joblib.load(CLINICAL_MODEL_PATH)
            print("app.py: Loaded clinical model.")
        except Exception as e:
            print(f"Error loading clinical model: {e}")
            
    if REVIEW_MODEL_PATH.exists():
        try:
            review_model = joblib.load(REVIEW_MODEL_PATH)
            print("app.py: Loaded review model.")
        except Exception as e:
            print(f"Error loading review model: {e}")
            
    # 2. Initialize RAG search engine
    try:
        rag_engine.initialize()
    except Exception as e:
        print(f"Error initializing RAG engine: {e}")
        
    # 3. Load PDF texts (for fallbacks)
    pdf_policy = PROJECT_ROOT / "data" / "raw" / "cms_mcd" / "clinical_policy_manual.pdf"
    if pdf_policy.exists():
        try:
            reader = pypdf.PdfReader(pdf_policy)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            CLINICAL_POLICY_TEXT = text
            print("app.py: Cached clinical policy text.")
        except Exception as e:
            print(f"Error caching clinical policy PDF text: {e}")
            
    pdf_handbook = PROJECT_ROOT / "data" / "raw" / "cms_mcd" / "member_handbook.pdf"
    if pdf_handbook.exists():
        try:
            reader = pypdf.PdfReader(pdf_handbook)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            MEMBER_HANDBOOK_TEXT = text
            print("app.py: Cached member handbook text.")
        except Exception as e:
            print(f"Error caching member handbook PDF text: {e}")

# Serve dashboard
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# Human Review Decision logging endpoint
@app.route("/api/prior-authorization/review", methods=["POST"])
def submit_human_review():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400
            
        request_id = str(data.get("request_id", "")).strip()
        reviewer_decision = str(data.get("reviewer_decision", "")).strip()
        review_reason = str(data.get("review_reason", "")).strip()
        
        if not request_id or not reviewer_decision:
            return jsonify({"error": "Missing request_id or reviewer_decision."}), 400
            
        updated_log = audit_logger.log_reviewer_decision(
            request_id=request_id,
            reviewer_decision=reviewer_decision,
            review_reason=review_reason
        )
        
        if updated_log:
            return jsonify({
                "status": "success",
                "message": f"Recorded human decision override: {reviewer_decision} for request {request_id}.",
                "audit_entry": updated_log
            })
        else:
            return jsonify({"error": f"Request ID {request_id} not found in audit trail."}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/prior-authorization/parse-pdf", methods=["POST"])
def parse_clinical_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400
        if file and file.filename.lower().endswith('.pdf'):
            from utils.pdf_parser import extract_text_from_pdf, parse_clinical_details
            text = extract_text_from_pdf(file)
            if not text:
                return jsonify({"error": "Could not extract text from PDF."}), 400
            details = parse_clinical_details(text)
            return jsonify(details)
        else:
            return jsonify({"error": "Only PDF files are supported."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Legacy endpoints mapping (to maintain backward compatibility with old codebases)
@app.route("/api/predict_clinical", methods=["POST"])
def predict_clinical_legacy():
    # Reroute to /api/prior-authorization/check logic but return in legacy format
    from api.check import check_prior_authorization
    return check_prior_authorization()

@app.route("/api/predict_review", methods=["POST"])
def predict_review_legacy():
    # Reroute to /api/prior-authorization/check logic but return in legacy format
    from api.check import check_prior_authorization
    return check_prior_authorization()

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Run initialization
initialize_application()

if __name__ == "__main__":
    # NOTE: debug=True must be set to False before any production deployment.
    # Werkzeug debug mode allows remote code execution if exposed.
    app.run(host="127.0.0.1", port=5000, debug=True)
