"""
HL7 FHIR R4 Bundle Exporter for Midtown Emergency Sheba.
Transforms AI Agent clinical outputs into international standard FHIR resources.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import time
import uuid

def generate_fhir_bundle(consultation_result):
    """
    Constructs an HL7 FHIR R4 Collection Bundle from the AI consultation output.
    """
    bundle_id = str(uuid.uuid4())
    patient_id = "patient-" + str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    patient_profile = consultation_result.get("patient_profile", {})
    triage = consultation_result.get("triage", {})
    soap = consultation_result.get("soap_note", {})
    meds = consultation_result.get("suggested_medications", [])
    doctors = consultation_result.get("recommended_doctors", [])

    # FHIR Patient Resource
    patient_resource = {
        "fullUrl": f"urn:uuid:{patient_id}",
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"],
                "source": "Midtown Emergency Sheba"
            },
            "name": [{"text": patient_profile.get("name", "Emergency Patient")}],
            "gender": (patient_profile.get("gender", "unknown")).lower()
        }
    }

    # FHIR Condition Resource (Diagnosis)
    condition_id = "condition-" + str(uuid.uuid4())[:8]
    condition_resource = {
        "fullUrl": f"urn:uuid:{condition_id}",
        "resource": {
            "resourceType": "Condition",
            "id": condition_id,
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active"
                }]
            },
            "verificationStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "provisional"
                }]
            },
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                    "code": "encounter-diagnosis",
                    "display": "Encounter Diagnosis"
                }]
            }],
            "severity": {
                "text": triage.get("priority", "URGENT")
            },
            "code": {
                "text": soap.get("assessment", "Acute Emergency")
            },
            "subject": {"reference": f"urn:uuid:{patient_id}"},
            "recordedDate": timestamp
        }
    }

    # FHIR Observation Resource (Triage & Vitals)
    observation_id = "obs-triage-" + str(uuid.uuid4())[:8]
    observation_resource = {
        "fullUrl": f"urn:uuid:{observation_id}",
        "resource": {
            "resourceType": "Observation",
            "id": observation_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                    "display": "Survey"
                }]
            }],
            "code": {
                "text": "Emergency Severity Index (ESI) Triage Classification"
            },
            "subject": {"reference": f"urn:uuid:{patient_id}"},
            "effectiveDateTime": timestamp,
            "valueInteger": triage.get("esi_level", 2),
            "note": [{"text": triage.get("description", "")}]
        }
    }

    entries = [patient_resource, condition_resource, observation_resource]

    # FHIR MedicationRequest Resources
    for i, med in enumerate(meds):
        med_req_id = f"medreq-{i+1}-" + str(uuid.uuid4())[:6]
        entries.append({
            "fullUrl": f"urn:uuid:{med_req_id}",
            "resource": {
                "resourceType": "MedicationRequest",
                "id": med_req_id,
                "status": "draft",
                "intent": "proposal",
                "medicationCodeableConcept": {
                    "text": med.get("name")
                },
                "subject": {"reference": f"urn:uuid:{patient_id}"},
                "dosageInstruction": [{
                    "text": med.get("dosage", "As directed")
                }],
                "note": [{"text": f"Safety warnings: {med.get('warnings', '')}"}]
            }
        })

    # FHIR CarePlan / Doctor referral
    if doctors:
        lead_doc = doctors[0]
        careplan_id = "careplan-" + str(uuid.uuid4())[:8]
        entries.append({
            "fullUrl": f"urn:uuid:{careplan_id}",
            "resource": {
                "resourceType": "CarePlan",
                "id": careplan_id,
                "status": "active",
                "intent": "plan",
                "title": f"Midtown Emergency Referral to {lead_doc['name']}",
                "description": f"Hospital: {lead_doc['hospital']}. Chamber: {lead_doc['chamber']}. Emergency Contact: {lead_doc['phone']}.",
                "subject": {"reference": f"urn:uuid:{patient_id}"}
            }
        })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "meta": {
            "lastUpdated": timestamp,
            "author": "Payal Rani Das, Midtown Eye Hospital, Comilla",
            "system": "Midtown Emergency Sheba Neural Protocol v2.5"
        },
        "type": "collection",
        "total": len(entries),
        "entry": entries
    }
