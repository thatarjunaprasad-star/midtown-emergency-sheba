# Midtown Emergency Sheba (মিডটাউন ইমার্জেন্সি সেবা)

> **High-Level Convolutional AI Medical & Emergency Agent**  
> **Creator:** Payal Rani Das, Midtown Eye Hospital, Comilla.  
> **Institutional Affiliation:** Midtown Eye Hospital, Laksham Road, Comilla, Bangladesh.  
> **Emergency Helpline:** `+880 1711-209988` | **National Emergency:** `999`

---

## 🌟 Overview

**Midtown Emergency Sheba** is a medical-grade, multimodal AI clinical triage and consultation system designed to provide real-time suggestions about **treatment**, **medicines**, and **specialist doctors**. It is specifically architected for ophthalmic and general emergency cases with a deep focus on **High-Level CNNs**, **RNNs / Bi-LSTMs**, and **LLM Clinical Reasoning Agents**, operating under standard front-end design protocols and hospital-grade back-end security.

---

## 🧠 Architectural Pillars

```
+---------------------------------------------------------------------------------------------------------+
|                                    MIDTOWN EMERGENCY SHEBA (ARCHITECTURE)                                |
+---------------------------------------------------------------------------------------------------------+
                                                     |
         +-------------------------------------------+------------------------------------------+
         |                                           |                                          |
         v                                           v                                          v
+-----------------------------+             +-----------------------------+            +----------------------------+
|     HIGH-LEVEL CNN LAB      |             |   SEQUENTIAL RNN / Bi-LSTM  |            |   MULTIMODAL LLM AGENT     |
|   (PyTorch MedicalCNN)      |             |   (ClinicalVitalsRNN)       |            |   (Clinical Reasoning)     |
+-----------------------------+             +-----------------------------+            +----------------------------+
| • 4 Deep Residual Blocks    |             | • 2-layer Bidirectional     |            | • ESI Triage Level (1 to 5)|
| • Grad-CAM Explainable AI   |             |   LSTM with Dropout         |            | • Structured SOAP Note     |
| • Ophthalmic & Trauma Foci: |             | • Temporal Attention Head   |            | • Bilingual Guidance       |
|   - Corneal Ulcer/Keratitis |             | • Time-series Vitals:       |            |   (English & বাংলা)        |
|   - Glaucoma Emergency      |             |   HR, SBP, DBP, SpO2, Temp, |            | • Contraindication Checks  |
|   - Diabetic Retinopathy    |             |   RR, Pain (0-10)           |            | • Pharmacopeia Validation  |
|   - Mature Cataract         |             | • Shock Index (HR/SBP)      |            | • Midtown Doctor Matching  |
|   - Chemical Burn Trauma    |             | • Acute Decline Probability |            | • Optional Gemini GenAI    |
+-----------------------------+             +-----------------------------+            +----------------------------+
         |                                           |                                          |
         +-------------------------------------------+------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                    HOSPITAL-GRADE SECURITY & COMPLIANCE                                 |
+---------------------------------------------------------------------------------------------------------+
| • AES-256-GCM Authenticated AEAD Encryption for Protected Health Information (PHI)                      |
| • SHA-256 Merkle-Chained Tamper-Evident Clinical Audit Ledger                                           |
| • HL7 FHIR R4 Standard Collection Bundles (Patient, Condition, Observation, MedicationRequest, CarePlan)|
| • Hospital Security Headers, Sliding-Window IP Rate Limiter & Sanitization                              |
+---------------------------------------------------------------------------------------------------------+
```

---

## 🔬 Core Components

### 1. High-Level CNN (PyTorch `MedicalCNN` with Grad-CAM)
- **Architecture:** 4-stage convolutional feature extractor with residual projections, batch normalization, ReLU, and global adaptive average pooling.
- **Explainable AI (Grad-CAM):** Computes gradients of the predicted pathology with respect to the `stage4` feature maps, generating real-time heatmaps that visually pinpoint lesion sites (e.g., corneal stromal infiltrates, microaneurysms, lens opacities).
- **Supported Pathologies:**
  1. *Corneal Ulcer / Severe Microbial Keratitis* (ESI Level 2)
  2. *Acute Angle-Closure Glaucoma Crisis* (ESI Level 1/2)
  3. *Diabetic Retinopathy Microvascular Disease* (ESI Level 3)
  4. *Mature Cataract* (ESI Level 4)
  5. *Acute Conjunctivitis / Hyperemia* (ESI Level 4)
  6. *Chemical Ocular Burn / Trauma* (ESI Level 1)
  7. *Normal Healthy Eye* (ESI Level 5)

### 2. High-Level RNN (PyTorch `ClinicalVitalsRNN`)
- **Architecture:** 2-layer Bidirectional LSTM (`input_dim=7, hidden_dim=64`) equipped with a custom **Temporal Attention** mechanism.
- **Sequential Physiological Tracking:** Tracks sequential vital signs over hours/intervals.
- **Computed Metrics:**
  - **Acute Deterioration Probability (%):** Predicts impending hemodynamic collapse.
  - **Shock Index ($HR / SBP$):** Normal ($0.5-0.7$), Shock Alert ($> 0.9$).
  - **Mean Arterial Pressure (MAP):** $\frac{2 \times DBP + SBP}{3}$.
  - **Temporal Attention Distribution:** Highlights which exact time step triggered clinical destabilization.

### 3. LLM Medical Clinical Reasoning Agent
- **Multimodal Synthesis:** Fuses patient complaints, CNN image findings, and RNN vitals trajectory.
- **Emergency Severity Index (ESI):** Automatically classifies cases from ESI-1 (Resuscitation) to ESI-5 (Non-urgent).
- **SOAP Format:** Standard Subjective, Objective, Assessment, and Plan documentation.
- **Bilingual Guidance:** Instant emergency first-aid advice in English and Bengali (বাংলা) with Text-to-Speech (TTS) audio narration.
- **Midtown Eye Hospital Doctor Matching:** Matches patients with lead consultants and emergency surgeons in Comilla, headed by **Dr. Payal Rani Das**.

### 4. Pharmacopeia & Interaction Safety Vault
- Comprehensive database of ophthalmic and emergency medications.
- Real-time **Drug-Drug Interaction Checker** (e.g., Timolol + Verapamil bradycardia warnings, Prednisolone + NSAID corneal perforation warnings).
- Strict contraindication guards (e.g., Asthma/COPD contraindications for Beta-blocker eye drops).

### 5. Hospital Security & Standard Protocol
- **AES-256-GCM AEAD:** Encrypts Protected Health Information (PHI) with 96-bit nonces.
- **Cryptographic Audit Ledger:** Merkle-style SHA-256 blockchain tracking every triage scan, consultation, and prescription.
- **HL7 FHIR R4 Bundle Standard:** One-click export into standard FHIR JSON for Electronic Medical Record (EMR) exchange.
- **Security Headers & Throttling:** Strict CSP, X-Frame-Options, X-Content-Type-Options, and sliding-window rate limiting.

---

## 👨‍⚕️ Midtown Eye Hospital, Comilla - Clinical Registry

| Doctor | Designation & Specialty | Chamber & Hours | Emergency On-Call | Contact |
|---|---|---|---|---|
| **Dr. Payal Rani Das** | Lead Consultant & Medical AI Director (Vitreo-Retina & Trauma Surgery) | Chamber 101, Executive Wing, Midtown Eye Hospital, Comilla (10AM-2PM, 5PM-9PM) | **YES (24/7)** | +880 1711-209988 |
| **Dr. M. A. Rahman** | Senior Consultant (Glaucoma & Anterior Segment) | Chamber 104, Midtown Eye Hospital, Comilla (4PM-9PM) | **YES** | +880 1712-334455 |
| **Dr. Farhana Sultana** | Consultant (Cornea, Keratitis & Chemical Burns) | Chamber 106, Midtown Eye Hospital, Comilla (9AM-1PM, 6PM-9PM) | **YES** | +880 1715-778899 |
| **Dr. Tanvir Ahmed** | Specialist (Pediatric Ophthalmology & Strabismus) | Chamber 108, Midtown Eye Hospital, Comilla (3PM-8PM) | Standard | +880 1718-445566 |
| **Dr. Ananya Sen** | Head of Emergency Triage & Casualty Care | Emergency ER Bay 1, Midtown Hospital Campus, Comilla | **YES (24/7)** | +880 1713-990011 |
| **Dr. Kamrul Hasan** | Senior Critical Care & Medicine Consultant | ICU Block A, Midtown Hospital, Comilla | **YES (24/7)** | +880 1714-882233 |

---

## 🚀 Quickstart & Running the App

### 1. Run Unit & Integration Tests
```powershell
python tests/test_models.py
python tests/test_security.py
python tests/test_api_endpoints.py
```

### 2. Start the Midtown Emergency Sheba Server
```powershell
python backend/run.py
```
Or with Uvicorn:
```powershell
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

### 3. Open in Browser
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern web browser to access the complete application interface.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health, model statuses, creator credentials |
| `GET` | `/api/samples` | List pre-generated ophthalmic sample images |
| `POST` | `/api/cnn/analyze` | Image upload / sample ID -> CNN inference + Grad-CAM heatmap |
| `POST` | `/api/rnn/analyze-vitals` | Sequential vitals array -> Bi-LSTM deterioration risk & shock index |
| `POST` | `/api/agent/consult` | Multimodal consult -> ESI triage, SOAP note, doctor matching, AES-256 encryption |
| `GET` | `/api/doctors` | List & filter doctors at Midtown Eye Hospital, Comilla |
| `GET` | `/api/medicines` | Search medications, indications, dosages |
| `POST` | `/api/medicines/interaction-check` | Check pairwise drug-drug interactions |
| `POST` | `/api/export/fhir` | Export consultation as standard HL7 FHIR R4 JSON bundle |
| `GET` | `/api/security/audit-trail` | Cryptographic SHA-256 audit ledger & integrity verification |

---

## 🎖️ Attribution & Credits

- **Application Name:** Midtown Emergency Sheba (মিডটাউন ইমার্জেন্সি সেবা)
- **Creator:** **Payal Rani Das, Midtown Eye Hospital, Comilla.**
- **Affiliation:** Midtown Eye Hospital, Laksham Road, Comilla, Bangladesh.
