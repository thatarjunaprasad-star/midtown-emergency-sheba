"""
FastAPI Server & REST API for Midtown Emergency Sheba.
Integrates High-level CNN, RNN, LLM Agent, Security Protocols, and Hospital Data.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import os
import io
import json
import base64
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Internal Modules
from backend.models.cnn_module import run_cnn_inference, CLASS_NAMES
from backend.models.rnn_module import run_rnn_inference
from backend.models.llm_agent import (
    run_llm_agent_consultation,
    check_drug_interactions,
    DOCTORS,
    MEDICINES
)
from backend.security.crypto import encrypt_phi, decrypt_phi, GLOBAL_AUDIT_LEDGER
from backend.security.middleware import SecurityHeadersMiddleware, RateLimiterMiddleware, sanitize_input_text
from backend.security.fhir import generate_fhir_bundle

# Initialize FastAPI App
app = FastAPI(
    title="Midtown Emergency Sheba",
    description="Convolutional AI Medical Agent by Payal Rani Das, Midtown Eye Hospital, Comilla.",
    version="2.5.0"
)

# Add Security Middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimiterMiddleware, max_requests_per_minute=180)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "sample_cases"))

# --- Pydantic Data Models ---
class VitalSignsRecord(BaseModel):
    time: str = "10:00"
    hr: float = Field(78.0, ge=30, le=250, description="Heart Rate in bpm")
    sbp: float = Field(120.0, ge=50, le=260, description="Systolic Blood Pressure")
    dbp: float = Field(80.0, ge=30, le=160, description="Diastolic Blood Pressure")
    spo2: float = Field(98.0, ge=50, le=100, description="Oxygen Saturation SpO2 %")
    temp: float = Field(37.0, ge=33, le=43, description="Body Temperature in Celsius")
    rr: float = Field(16.0, ge=6, le=60, description="Respiratory Rate")
    pain: float = Field(2.0, ge=0, le=10, description="Pain Score 0 to 10")

class VitalsTrajectoryRequest(BaseModel):
    patient_id: Optional[str] = "P-101"
    readings: List[VitalSignsRecord]

class ConsultationRequest(BaseModel):
    patient_name: str = "Rahim Ahmed"
    age: int = 48
    gender: str = "Male"
    chief_complaint: str
    medical_history: Optional[str] = ""
    cnn_result: Optional[Dict[str, Any]] = None
    rnn_result: Optional[Dict[str, Any]] = None
    language: str = "both"

class DrugInteractionRequest(BaseModel):
    drug_names: List[str]

# --- REST Endpoints ---

@app.get("/api/status")
async def get_system_status():
    """
    Returns system status, active neural models, and creator attribution.
    """
    return {
        "status": "ONLINE",
        "app_name": "Midtown Emergency Sheba (মিডটাউন ইমার্জেন্সি সেবা)",
        "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
        "hospital": "Midtown Eye Hospital, Comilla, Bangladesh",
        "emergency_hotline": "+880 1711-209988",
        "architectures": {
            "cnn": "Deep Residual Convolutional Neural Network with Grad-CAM Explainable Attention",
            "rnn": "Bidirectional LSTM with Temporal Attention for Sequential Patient Vitals",
            "llm_agent": "Multimodal Medical Reasoning Agent & Triage Orchestrator"
        },
        "security_protocols": {
            "encryption": "AES-256-GCM for Protected Health Information (PHI)",
            "audit_trail": "SHA-256 Cryptographic Tamper-evident Ledger",
            "standards": ["HL7 FHIR R4", "Emergency Severity Index (ESI Level 1-5)", "HIPAA Guidelines"]
        }
    }


@app.get("/api/samples")
async def get_sample_images():
    """
    Returns available clinical sample images for instant one-click testing.
    """
    samples = []
    if os.path.exists(SAMPLE_DIR):
        for f in os.listdir(SAMPLE_DIR):
            if f.endswith((".jpg", ".png")):
                key = os.path.splitext(f)[0]
                samples.append({
                    "sample_id": key,
                    "filename": f,
                    "title": key.replace("_", " ").title(),
                    "download_url": f"/api/samples/download/{f}"
                })
    return {"samples": samples}


@app.get("/api/samples/download/{filename}")
async def download_sample(filename: str):
    path = os.path.join(SAMPLE_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Sample image not found")


@app.post("/api/cnn/analyze")
async def analyze_medical_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None)
):
    """
    Runs High-level CNN visual inference with Grad-CAM activation heatmap generation.
    Supports either direct file upload or sample preset ID.
    """
    image_bytes = None
    
    if sample_id:
        sample_path = os.path.join(SAMPLE_DIR, f"{sample_id}.jpg")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
        else:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
    elif file:
        image_bytes = await file.read()
    else:
        raise HTTPException(status_code=400, detail="Must provide an uploaded image file or a sample_id.")

    try:
        report = run_cnn_inference(image_bytes)
        
        # Record tamper-evident audit block
        GLOBAL_AUDIT_LEDGER.record_event(
            event_type="CNN_VISION_INFERENCE",
            data_summary=f"Detected {report['predicted_class']} (Confidence: {report['confidence_percent']}%, Urgency: {report['urgency']})"
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CNN Vision Processing Error: {str(e)}")


@app.post("/api/rnn/analyze-vitals")
async def analyze_vitals_trajectory(req: VitalsTrajectoryRequest):
    """
    Runs High-level RNN / Bi-LSTM inference on sequential vitals time series.
    """
    readings_dict = [r.model_dump() for r in req.readings]
    try:
        result = run_rnn_inference(readings_dict)
        
        # Record audit block
        GLOBAL_AUDIT_LEDGER.record_event(
            event_type="RNN_VITALS_TRAJECTORY",
            data_summary=f"Patient {req.patient_id} - Trajectory: {result['trajectory_status']} (Deterioration Risk: {result['deterioration_risk_percent']}%, Shock Index: {result['shock_index']})"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RNN Vitals Inference Error: {str(e)}")


@app.post("/api/agent/consult")
async def agent_consult(req: ConsultationRequest):
    """
    Multimodal AI Medical Consultation:
    Integrates complaints, history, CNN findings, and RNN vitals trajectory.
    Applies standard triage protocol, safety verification, and AES-256 encryption.
    """
    # Sanitize user inputs
    clean_complaint = sanitize_input_text(req.chief_complaint)
    clean_history = sanitize_input_text(req.medical_history or "")
    clean_name = sanitize_input_text(req.patient_name)

    consult_result = run_llm_agent_consultation(
        patient_name=clean_name,
        age=req.age,
        gender=req.gender,
        chief_complaint=clean_complaint,
        medical_history=clean_history,
        cnn_result=req.cnn_result,
        rnn_result=req.rnn_result,
        language=req.language
    )

    # AES-256 Protected Health Information (PHI) Encryption Demonstration
    phi_payload = {
        "patient_name": clean_name,
        "age": req.age,
        "gender": req.gender,
        "chief_complaint": clean_complaint,
        "triage_level": consult_result["triage"]["esi_level"],
        "primary_plan": consult_result["soap_note"]["plan"]
    }
    encrypted_phi = encrypt_phi(phi_payload)

    # Cryptographic Audit Ledger Entry
    audit_entry = GLOBAL_AUDIT_LEDGER.record_event(
        event_type="AI_CLINICAL_CONSULTATION",
        data_summary=f"Consultation for {clean_name} - ESI: {consult_result['triage']['esi_badge']} - Creator: Payal Rani Das, Midtown Eye Hospital, Comilla"
    )

    consult_result["security"] = {
        "phi_encryption": encrypted_phi,
        "audit_block_hash": audit_entry["hash"],
        "audit_block_index": audit_entry["index"],
        "compliance": "HIPAA & HL7 FHIR R4 Ready"
    }

    return consult_result


@app.get("/api/doctors")
async def list_doctors(specialty: Optional[str] = Query(None), emergency_only: bool = False):
    """
    Returns list of doctors at Midtown Eye Hospital, Comilla with optional filtering.
    """
    results = DOCTORS
    if emergency_only:
        results = [d for d in results if d.get("emergency_on_call")]
    if specialty:
        results = [d for d in results if specialty.lower() in d["specialty"].lower()]
    return {
        "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
        "hospital": "Midtown Eye Hospital, Comilla",
        "total": len(results),
        "doctors": results
    }


@app.get("/api/medicines")
async def list_medicines(query: Optional[str] = Query(None)):
    """
    Search medicines, ophthalmic eye drops, and emergency drugs.
    """
    if not query:
        return {"medicines": MEDICINES}
    q = query.lower()
    matches = [
        m for m in MEDICINES
        if q in m["name"].lower() or any(q in b.lower() for b in m.get("brand_names", [])) or q in m["category"].lower()
    ]
    return {"query": query, "count": len(matches), "medicines": matches}


@app.post("/api/medicines/interaction-check")
async def check_interactions(req: DrugInteractionRequest):
    """
    Evaluates drug interaction safety rules.
    """
    interactions = check_drug_interactions(req.drug_names)
    return {
        "evaluated_drugs": req.drug_names,
        "has_interactions": len(interactions) > 0,
        "interaction_count": len(interactions),
        "interactions": interactions
    }


@app.post("/api/export/fhir")
async def export_fhir(consultation_data: Dict[str, Any]):
    """
    Generates an HL7 FHIR R4 standard JSON Bundle for hospital EMR integration.
    """
    fhir_bundle = generate_fhir_bundle(consultation_data)
    return fhir_bundle


@app.get("/api/security/audit-trail")
async def get_audit_trail():
    """
    Returns the cryptographic SHA-256 audit ledger and verifies mathematical integrity.
    """
    is_valid, message = GLOBAL_AUDIT_LEDGER.verify_integrity()
    return {
        "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
        "ledger_blocks_count": len(GLOBAL_AUDIT_LEDGER.chain),
        "mathematical_integrity_verified": is_valid,
        "verification_message": message,
        "blocks": GLOBAL_AUDIT_LEDGER.chain
    }


# Mount Static Frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
