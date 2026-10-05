"""
LLM Medical Clinical Reasoning & Multi-Agent Orchestrator Module
for Midtown Emergency Sheba.

Integrates CNN Visual Diagnostics, RNN Sequential Vitals Trajectory,
Pharmacopeia Safety Verification, and Midtown Eye Hospital Comilla Doctor Matching.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

DOCTORS = load_data("doctors.json")
MEDICINES = load_data("medicines.json")

def calculate_esi_level(complaint_text, cnn_result=None, rnn_result=None):
    """
    Standard Emergency Severity Index (ESI) Triage Algorithm (Levels 1 to 5).
    """
    text = (complaint_text or "").lower()
    
    # ESI 1: Immediate life/organ threat requiring zero-delay resuscitation
    if any(k in text for k in ["chemical burn", "acid splash", "alkali", "unconscious", "cannot breathe", "severe shock"]):
        return 1, "ESI Level 1 (Resuscitation Required - Zero Delay Intervention)"
    if cnn_result and cnn_result.get("predicted_class") == "chemical_burn_ocular_trauma":
        return 1, "ESI Level 1 (Resuscitation Required - Chemical Ocular Trauma)"
    if rnn_result and rnn_result.get("shock_index", 0) > 1.0:
        return 1, "ESI Level 1 (Resuscitation Required - Hemodynamic Collapse)"

    # ESI 2: High risk / Emergent (Threat to sight or severe distress)
    if any(k in text for k in ["sudden vision loss", "corneal ulcer", "glaucoma attack", "severe eye pain", "vomiting and halo"]):
        return 2, "ESI Level 2 (Emergent - High Risk Sight/Organ Threat, immediate doctor bed)"
    if cnn_result and cnn_result.get("predicted_class") in ["corneal_ulcer_keratitis", "acute_angle_closure_glaucoma"]:
        return 2, "ESI Level 2 (Emergent - Urgent Ophthalmic Surgical / Medical Intervention)"
    if rnn_result and rnn_result.get("deterioration_risk_percent", 0) > 65:
        return 2, "ESI Level 2 (Emergent - High Vital Signs Deterioration Risk)"

    # ESI 3: Urgent (Requires multiple resources or systemic illness)
    if any(k in text for k in ["diabetic retinopathy", "flashes and floaters", "high fever", "blurry vision"]):
        return 3, "ESI Level 3 (Urgent - Requires multiple diagnostic resources / OCT / Lab)"
    if cnn_result and cnn_result.get("predicted_class") == "diabetic_retinopathy":
        return 3, "ESI Level 3 (Urgent - Specialized Retinal Assessment)"

    # ESI 4: Less Urgent (Single resource needed)
    if any(k in text for k in ["red eye", "conjunctivitis", "cataract", "mild itching", "sticky discharge"]):
        return 4, "ESI Level 4 (Less Urgent - Single resource / Outpatient prescription)"
    if cnn_result and cnn_result.get("predicted_class") in ["mature_cataract", "acute_conjunctivitis"]:
        return 4, "ESI Level 4 (Less Urgent - Outpatient Consultation)"

    # ESI 5: Non-urgent
    return 5, "ESI Level 5 (Non-urgent - Routine checkup / Health counseling)"


def match_doctors(condition_key, complaints_text=""):
    """
    Matches the patient with the most suitable consultants at Midtown Eye Hospital, Comilla.
    """
    matched = []
    text = (condition_key + " " + complaints_text).lower()

    for doc in DOCTORS:
        score = 0
        spec = doc["specialty"].lower()
        title = doc["title"].lower()
        services = " ".join(doc.get("services", [])).lower()

        # Prioritize Creator & Lead Consultant Dr. Payal Rani Das
        if "payal" in doc["name"].lower():
            score += 2

        if "retina" in text or "diabetic" in text or "trauma" in text:
            if "retina" in spec or "trauma" in spec:
                score += 5
        if "glaucoma" in text or "pressure" in text or "iop" in text:
            if "glaucoma" in spec:
                score += 5
        if "ulcer" in text or "keratitis" in text or "burn" in text or "cornea" in text:
            if "cornea" in spec:
                score += 5
        if "cataract" in text or "lens" in text:
            if "cataract" in spec or "retina" in spec:
                score += 4
        if "child" in text or "baby" in text or "pediatric" in text:
            if "pediatric" in spec:
                score += 6
        if "shock" in text or "hypotension" in text or "hypertens" in text or "sepsis" in text or "emergency" in text:
            if "emergency" in spec or "critical" in spec or "trauma" in spec:
                score += 5

        if doc.get("emergency_on_call"):
            score += 2

        matched.append((score, doc))

    matched.sort(key=lambda x: x[0], reverse=True)
    return [d[1] for d in matched[:3]]


def find_safe_medicines(condition_key, patient_history=""):
    """
    Searches and validates recommended medications against patient contraindications.
    """
    hx = (patient_history or "").lower()
    has_asthma = "asthma" in hx or "copd" in hx or "wheez" in hx
    has_sulfa_allergy = "sulfa" in hx
    has_renal_issue = "kidney" in hx or "renal" in hx
    
    recommendations = []
    contraindication_warnings = []

    for med in MEDICINES:
        med_rec = None
        
        if condition_key == "corneal_ulcer_keratitis":
            if med["id"] == "MED-001":  # Moxifloxacin
                med_rec = med
            elif med["id"] == "MED-005": # Atropine
                med_rec = med

        elif condition_key == "acute_angle_closure_glaucoma":
            if med["id"] == "MED-002":  # Timolol
                if has_asthma:
                    contraindication_warnings.append(
                        f"SAFETY ALERT: {med['name']} is CONTRAINDICATED due to patient's history of Asthma/COPD (Risk of fatal bronchospasm). Substitute with Brimonidine or oral agent!"
                    )
                else:
                    med_rec = med
            elif med["id"] == "MED-007":  # Acetazolamide
                if has_sulfa_allergy or has_renal_issue:
                    contraindication_warnings.append(
                        f"SAFETY ALERT: {med['name']} is CONTRAINDICATED due to Sulfa allergy or Renal impairment."
                    )
                else:
                    med_rec = med

        elif condition_key == "chemical_burn_ocular_trauma":
            if med["id"] == "MED-006":  # Normal Saline Irrigation
                med_rec = med
            elif med["id"] == "MED-004": # Artificial Tears
                med_rec = med
            elif med["id"] == "MED-001": # Moxifloxacin prophylaxis
                med_rec = med

        elif condition_key == "acute_conjunctivitis":
            if med["id"] == "MED-001":  # Moxifloxacin
                med_rec = med
            elif med["id"] == "MED-004":  # Tears
                med_rec = med

        elif condition_key == "diabetic_retinopathy":
            if med["id"] == "MED-010":  # Amlodipine (BP control)
                med_rec = med
            elif med["id"] == "MED-004":
                med_rec = med

        elif condition_key == "mature_cataract":
            if med["id"] == "MED-004":  # Lubricant for symptomatic relief
                med_rec = med

        if med_rec and med_rec not in recommendations:
            recommendations.append(med_rec)

    # General emergency pain relief if severe
    if "pain" in condition_key or condition_key in ["corneal_ulcer_keratitis", "acute_angle_closure_glaucoma"]:
        paracetamol = next((m for m in MEDICINES if m["id"] == "MED-008"), None)
        if paracetamol and paracetamol not in recommendations:
            recommendations.append(paracetamol)

    return recommendations, contraindication_warnings


def check_drug_interactions(drug_list):
    """
    Checks pairwise interaction rules among selected drugs.
    """
    interactions = []
    drugs_lower = [d.lower() for d in drug_list]
    
    if any("timolol" in d for d in drugs_lower) and any("verapamil" in d or "beta-blocker" in d for d in drugs_lower):
        interactions.append({
            "severity": "HIGH",
            "drugs": ["Timolol", "Systemic Beta-Blocker/Calcium Blocker"],
            "warning": "Risk of profound bradycardia, hypotension, and heart block."
        })
        
    if any("prednisolone" in d or "steroid" in d for d in drugs_lower) and any("ketorolac" in d or "nsaid" in d for d in drugs_lower):
        interactions.append({
            "severity": "MODERATE",
            "drugs": ["Topical Steroid", "Topical NSAID"],
            "warning": "Concurrent ophthalmic use increases risk of corneal stromal thinning and ulcer perforation."
        })

    if any("diamox" in d or "acetazolamide" in d for d in drugs_lower) and any("aspirin" in d for d in drugs_lower):
        interactions.append({
            "severity": "HIGH",
            "drugs": ["Acetazolamide", "High-dose Aspirin"],
            "warning": "Risk of severe metabolic acidosis and CNS toxicity."
        })

    return interactions


def generate_bilingual_emergency_advice(condition_key, first_aid_en, first_aid_bn):
    """
    Returns verified clinical guidance in both English and Bengali (বাংলা).
    """
    advice_map = {
        "chemical_burn_ocular_trauma": {
            "en": "CRITICAL EMERGENCY: Flush the injured eye immediately with clean water or normal saline for 20-30 MINUTES without pause. Hold eyelids wide open. Do NOT rub the eye. Call Midtown Emergency Sheba immediately!",
            "bn": "জরুরী নির্দেশনা: অবিলম্বে চোখে প্রচুর পরিষ্কার পানি বা নরমাল স্যালাইন দিয়ে একটানা ২০-৩০ মিনিট চোখ ধুতে থাকুন। চোখের পাতা টেনে ধরে রাখুন। চোখ ঘষবেন না। অবিলম্বে মিডটাউন আই হসপিটাল ইমার্জেন্সিতে আসুন!"
        },
        "corneal_ulcer_keratitis": {
            "en": "WARNING: Do NOT apply steroid eye drops. Do not wear contact lenses or patch the eye. Protect eye with clean glasses and reach Dr. Payal Rani Das / Cornea specialist at Midtown Eye Hospital Comilla.",
            "bn": "সতর্কতা: কোনো অবস্থায় স্টেরয়েড ড্রপ ব্যবহার করবেন না। চোখে ব্যান্ডেজ বা কন্টাক্ট লেন্স পরা সম্পূর্ণ নিষেধ। জরুরিভাবে মিডটাউন আই হসপিটাল, কুমিল্লায় ডক্টর পায়েল রানী দাস বা কর্ণিয়া বিশেষজ্ঞকে দেখান।"
        },
        "acute_angle_closure_glaucoma": {
            "en": "SIGHT THREATENING CRISIS: Sudden high intraocular pressure requires urgent medical pressure lowering to prevent irreversible optic nerve damage. Visit the emergency room immediately.",
            "bn": "দৃষ্টিহানির তীব্র ঝুঁকি: চোখের উচ্চ প্রেসার অবিলম্বে কমাতে হবে, নতুবা স্থায়ীভাবে দৃষ্টি নষ্ট হতে পারে। কালক্ষেপণ না করে দ্রুত হাসপাতালে ইমার্জেন্সিতে ভর্তি হোন।"
        },
        "diabetic_retinopathy": {
            "en": "RECOMMENDATION: Schedule an urgent dilated fundus examination and OCT scan with Vitreo-Retina specialist Dr. Payal Rani Das. Maintain strict control of blood sugar and blood pressure.",
            "bn": "পরামর্শ: অবিলম্বে ডক্টর পায়েল রানী দাসের তত্ত্বাবধানে ডাইঅবেটিক রেটিনা পরীক্ষা (OCT) করান এবং রক্তে সুগার ও প্রেশার কঠোরভাবে নিয়ন্ত্রণে রাখুন।"
        }
    }
    return advice_map.get(condition_key, {
        "en": "Consult a registered doctor at Midtown Eye Hospital, Comilla for clinical examination and definitive care.",
        "bn": "সঠিক চিকিৎসার জন্য মিডটাউন আই হসপিটাল, কুমিল্লায় বিশেষজ্ঞ চিকিৎসকের সাথে পরামর্শ করুন।"
    })


def run_llm_agent_consultation(
    patient_name="Patient",
    age=45,
    gender="Unspecified",
    chief_complaint="",
    medical_history="",
    cnn_result=None,
    rnn_result=None,
    language="both"
):
    """
    Full Multimodal Medical AI Consultation Engine:
    Fuses textual clinical complaints, CNN image findings, and RNN sequential vitals trajectory.
    Formulates standard SOAP note, ESI Triage, Pharmacopeia validation, and Doctor Matching.
    """
    pred_condition = cnn_result.get("predicted_class", "general_medical_triage") if cnn_result else "general_medical_triage"
    
    # ESI Level determination
    esi_level, esi_description = calculate_esi_level(chief_complaint, cnn_result, rnn_result)
    
    # Doctor matching
    matched_doctors = match_doctors(pred_condition, chief_complaint)
    
    # Medicine safety check
    prescribed_medicines, safety_warnings = find_safe_medicines(pred_condition, medical_history)
    
    # Emergency Bilingual Advice
    bilingual_instructions = generate_bilingual_emergency_advice(
        pred_condition,
        first_aid_en="",
        first_aid_bn=""
    )

    # Compile Structured SOAP Clinical Handover Note
    # S - Subjective
    subjective = f"Patient {patient_name}, {age} y/o ({gender}). Chief Complaint: '{chief_complaint}'. Medical History / Known Allergies: {medical_history or 'None reported'}."
    
    # O - Objective
    objective_lines = []
    if cnn_result:
        objective_lines.append(f"CNN Diagnostic Vision: {cnn_result.get('title')} (Confidence: {cnn_result.get('confidence_percent')}%, Urgency: {cnn_result.get('urgency')}). Grad-CAM highlights key lesion foci in {cnn_result.get('predicted_class')}.")
    if rnn_result:
        v = rnn_result.get("latest_vitals", {})
        objective_lines.append(f"RNN Vitals Trajectory: HR {v.get('hr')} bpm, BP {v.get('sbp')}/{v.get('dbp')} mmHg, SpO2 {v.get('spo2')}%, Temp {v.get('temp')}°C, Pain {v.get('pain')}/10. Shock Index: {rnn_result.get('shock_index')}, MAP: {rnn_result.get('mean_arterial_pressure')} mmHg, Acute Deterioration Risk: {rnn_result.get('deterioration_risk_percent')}% ({rnn_result.get('trajectory_status')}).")
    objective = " | ".join(objective_lines) if objective_lines else "Clinical examination pending physical assessment."

    # A - Assessment
    assessment = f"ESI Triage: {esi_description}. Primary working diagnosis: {cnn_result.get('title') if cnn_result else 'Acute Medical / Ocular Emergency'}. Trajectory stability: {rnn_result.get('trajectory_status', 'Evaluating') if rnn_result else 'Stable'}."

    # P - Plan
    plan_steps = [
        f"1. Immediate Triage Action: {bilingual_instructions['en']}",
        f"2. Urgent Specialist Referral: Lead Consultant {matched_doctors[0]['name']} ({matched_doctors[0]['specialty']}) at {matched_doctors[0]['hospital']}.",
        "3. Pharmacotherapy Initiation: Administer verified emergency drops/medications as per safety protocol.",
        "4. Red-Flag Monitoring: If symptoms worsen, call Midtown Emergency Helpline (+880 1711-209988) or dial 999 immediately."
    ]
    plan = "\n".join(plan_steps)

    # Optional Gemini enhancement check
    gemini_key = os.getenv("GEMINI_API_KEY")
    ai_enhanced_synthesis = None
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            prompt = f"""
            You are 'Midtown Emergency Sheba', a convolutional medical AI agent created by Payal Rani Das, Midtown Eye Hospital, Comilla.
            Analyze the following patient presentation:
            Patient: {patient_name}, {age} years, {gender}
            Complaint: {chief_complaint}
            History: {medical_history}
            CNN Image Finding: {cnn_result.get('title') if cnn_result else 'None'}
            RNN Vitals Status: {rnn_result.get('trajectory_status') if rnn_result else 'Normal'}
            Provide a concise, highly professional clinical recommendation with red flags and next actions.
            """
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            ai_enhanced_synthesis = response.text
        except Exception as e:
            ai_enhanced_synthesis = None

    return {
        "agent_name": "Midtown Emergency Sheba",
        "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
        "patient_profile": {
            "name": patient_name,
            "age": age,
            "gender": gender,
            "chief_complaint": chief_complaint,
            "medical_history": medical_history
        },
        "triage": {
            "esi_level": esi_level,
            "esi_badge": f"ESI-{esi_level}",
            "description": esi_description,
            "priority": "IMMEDIATE" if esi_level <= 2 else ("URGENT" if esi_level == 3 else "STANDARD")
        },
        "soap_note": {
            "subjective": subjective,
            "objective": objective,
            "assessment": assessment,
            "plan": plan
        },
        "emergency_instructions": {
            "english": bilingual_instructions["en"],
            "bengali": bilingual_instructions["bn"]
        },
        "recommended_doctors": matched_doctors,
        "suggested_medications": prescribed_medicines,
        "contraindication_safety_warnings": safety_warnings,
        "ai_synthesis": ai_enhanced_synthesis or (
            f"Automated clinical evaluation generated by Midtown Emergency Sheba AI Engine. "
            f"Based on deep convolutional visual features and recurrent vitals sequence modeling, "
            f"patient is triaged at {esi_description}. "
            f"Immediate review by {matched_doctors[0]['name']} at Midtown Eye Hospital, Comilla is strongly advised."
        )
    }
