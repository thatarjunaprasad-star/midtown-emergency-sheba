"""
Integration & End-to-End API endpoint tests for Midtown Emergency Sheba.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import os
import sys
from starlette.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.api import app

client = TestClient(app)

def test_api_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert data["creator"] == "Payal Rani Das, Midtown Eye Hospital, Comilla"
    print("Test /api/status Passed!")

def test_api_samples():
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) > 0
    print(f"Test /api/samples Passed! Found {len(data['samples'])} samples.")

def test_api_cnn_inference():
    res = client.post("/api/cnn/analyze", data={"sample_id": "corneal_ulcer"})
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert "visual_explanation" in data
    assert data["creator"] == "Payal Rani Das, Midtown Eye Hospital, Comilla"
    print("Test /api/cnn/analyze with sample Passed!")

def test_api_rnn_vitals():
    payload = {
        "patient_id": "PT-TEST-01",
        "readings": [
            {"time": "08:00", "hr": 78, "sbp": 120, "dbp": 80, "spo2": 98, "temp": 37.0, "rr": 16, "pain": 2},
            {"time": "10:00", "hr": 105, "sbp": 140, "dbp": 90, "spo2": 95, "temp": 38.0, "rr": 20, "pain": 6}
        ]
    }
    res = client.post("/api/rnn/analyze-vitals", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "deterioration_risk_percent" in data
    assert "shock_index" in data
    print("Test /api/rnn/analyze-vitals Passed!")

def test_api_agent_consult():
    payload = {
        "patient_name": "Afia Begum",
        "age": 42,
        "gender": "Female",
        "chief_complaint": "Sudden severe pain in left eye, blurred vision, colored halos around lights",
        "medical_history": "None known"
    }
    res = client.post("/api/agent/consult", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["creator"] == "Payal Rani Das, Midtown Eye Hospital, Comilla"
    assert "triage" in data
    assert "soap_note" in data
    assert "security" in data
    assert "audit_block_hash" in data["security"]
    print("Test /api/agent/consult Passed!")

def test_api_doctors_and_medicines():
    res_docs = client.get("/api/doctors")
    assert res_docs.status_code == 200
    assert len(res_docs.json()["doctors"]) >= 6

    res_meds = client.get("/api/medicines?query=Timolol")
    assert res_meds.status_code == 200
    assert res_meds.json()["count"] >= 1

    res_inter = client.post("/api/medicines/interaction-check", json={"drug_names": ["Timolol", "Verapamil"]})
    assert res_inter.status_code == 200
    assert res_inter.json()["has_interactions"] is True
    print("Test Doctors & Medicines & Interactions Passed!")

def test_api_security_audit_and_fhir():
    res_audit = client.get("/api/security/audit-trail")
    assert res_audit.status_code == 200
    assert res_audit.json()["mathematical_integrity_verified"] is True

    dummy_consult = {
        "patient_profile": {"name": "Test Patient", "gender": "Female"},
        "triage": {"esi_level": 2, "description": "ESI-2", "priority": "EMERGENT"},
        "soap_note": {"assessment": "Ocular Emergency", "plan": "Immediate exam"},
        "suggested_medications": [{"name": "Moxifloxacin", "dosage": "1 drop q2h"}],
        "recommended_doctors": [{"name": "Dr. Payal Rani Das", "hospital": "Midtown Eye Hospital, Comilla", "phone": "+880 1711-209988", "chamber": "Chamber 101"}]
    }
    res_fhir = client.post("/api/export/fhir", json=dummy_consult)
    assert res_fhir.status_code == 200
    assert res_fhir.json()["resourceType"] == "Bundle"
    print("Test Security Audit & FHIR Export Passed!")

def test_frontend_serving():
    res = client.get("/")
    assert res.status_code == 200
    assert "Midtown Emergency Sheba" in res.text
    assert "Payal Rani Das, Midtown Eye Hospital, Comilla" in res.text
    print("Test Frontend Static Serving Passed!")

if __name__ == "__main__":
    test_api_status()
    test_api_samples()
    test_api_cnn_inference()
    test_api_rnn_vitals()
    test_api_agent_consult()
    test_api_doctors_and_medicines()
    test_api_security_audit_and_fhir()
    test_frontend_serving()
    print("\nALL ENDPOINT & FRONTEND INTEGRATION TESTS PASSED 100%!")
