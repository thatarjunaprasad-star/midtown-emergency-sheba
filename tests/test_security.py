"""
Security, Encryption, and FHIR Compliance tests for Midtown Emergency Sheba.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.security.crypto import encrypt_phi, decrypt_phi, AuditLedger
from backend.security.fhir import generate_fhir_bundle

def test_aes256_gcm_phi_encryption():
    patient_data = {
        "name": "Fatema Begum",
        "phone": "+880 1819-123456",
        "condition": "Acute Angle Closure Glaucoma",
        "iop_mmhg": 48
    }

    encrypted = encrypt_phi(patient_data)
    assert encrypted["cipher_algorithm"] == "AES-256-GCM"
    assert "ciphertext" in encrypted
    assert "nonce" in encrypted

    # Decrypt and compare
    decrypted = decrypt_phi(encrypted)
    assert decrypted["name"] == patient_data["name"]
    assert decrypted["iop_mmhg"] == 48
    print("AES-256-GCM PHI Encryption & Decryption Passed!")

def test_tamper_evident_audit_ledger():
    ledger = AuditLedger()
    b1 = ledger.record_event("TRIAGE_ASSESSMENT", "Patient P-001 triaged to ESI-2")
    b2 = ledger.record_event("MEDICATION_PRESCRIBED", "Moxifloxacin eye drops 0.5% prescribed")

    is_valid, msg = ledger.verify_integrity()
    assert is_valid is True
    print(f"Ledger Verification Passed: {msg}")

    # Tamper test
    ledger.chain[1]["data_summary"] = "HACKED_DATA_SUMMARY"
    tampered_valid, tamper_msg = ledger.verify_integrity()
    assert tampered_valid is False
    print(f"Tamper Detection Succeeded: {tamper_msg}")

def test_fhir_bundle_generation():
    dummy_consult = {
        "patient_profile": {"name": "Kamal Hossain", "gender": "male"},
        "triage": {"esi_level": 2, "description": "ESI Level 2 (Emergent)", "priority": "EMERGENT"},
        "soap_note": {
            "subjective": "Severe left eye redness and pain",
            "objective": "Corneal ulcer",
            "assessment": "Corneal Ulcer",
            "plan": "Topical fortified moxifloxacin"
        },
        "suggested_medications": [{"name": "Moxifloxacin 0.5%", "dosage": "1 drop q1h"}],
        "recommended_doctors": [{
            "name": "Dr. Payal Rani Das",
            "hospital": "Midtown Eye Hospital, Comilla",
            "chamber": "Chamber 101, Executive Wing",
            "phone": "+880 1711-209988"
        }]
    }

    bundle = generate_fhir_bundle(dummy_consult)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["total"] >= 4
    res_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Patient" in res_types
    assert "Condition" in res_types
    assert "Observation" in res_types
    assert "MedicationRequest" in res_types
    assert "CarePlan" in res_types
    print("HL7 FHIR R4 Bundle Validation Passed!")

if __name__ == "__main__":
    test_aes256_gcm_phi_encryption()
    test_tamper_evident_audit_ledger()
    test_fhir_bundle_generation()
    print("ALL SECURITY & PROTOCOL TESTS PASSED SUCCESSFULLY!")
