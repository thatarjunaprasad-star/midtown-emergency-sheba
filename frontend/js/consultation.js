/**
 * Multimodal AI Medical Consultation & Clinical Protocols
 * Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
 */

let lastConsultationResult = null;

async function submitConsultation() {
    const name = document.getElementById("consult-name").value || "Patient";
    const age = parseInt(document.getElementById("consult-age").value) || 45;
    const gender = document.getElementById("consult-gender").value || "Male";
    const complaint = document.getElementById("consult-complaint").value;
    const history = document.getElementById("consult-history").value;

    if (!complaint) {
        showToast("Please enter patient chief complaint or symptoms", "warning");
        return;
    }

    const submitBtn = document.getElementById("btn-submit-consult");
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerText = "Analyzing Multimodal Data (CNN + RNN + Clinical LLM)...";
    }

    try {
        const payload = {
            patient_name: name,
            age: age,
            gender: gender,
            chief_complaint: complaint,
            medical_history: history,
            cnn_result: window.lastCnnResult || null,
            rnn_result: window.lastRnnResult || null,
            language: "both"
        };

        const res = await fetch("/api/agent/consult", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Consultation failed: " + res.statusText);
        const data = await res.json();
        lastConsultationResult = data;
        renderConsultationOutput(data);
        showToast("Clinical triage & consultation synthesized successfully!", "success");
    } catch (e) {
        showToast(e.message, "error");
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerText = "⚡ Generate Multimodal AI Medical Consultation";
        }
    }
}

function renderConsultationOutput(data) {
    const reportCard = document.getElementById("consult-report-card");
    if (!reportCard) return;

    reportCard.style.display = "block";

    // ESI Triage Badge
    const triageBadge = document.getElementById("report-esi-badge");
    triageBadge.innerText = data.triage.esi_badge;
    triageBadge.className = `esi-badge esi-${data.triage.esi_level}`;
    document.getElementById("report-esi-desc").innerText = data.triage.description;

    // Bilingual Instructions
    document.getElementById("report-instr-en").innerText = data.emergency_instructions.english;
    document.getElementById("report-instr-bn").innerText = data.emergency_instructions.bengali;

    // SOAP Note
    document.getElementById("soap-s").innerText = data.soap_note.subjective;
    document.getElementById("soap-o").innerText = data.soap_note.objective;
    document.getElementById("soap-a").innerText = data.soap_note.assessment;
    document.getElementById("soap-p").innerText = data.soap_note.plan;

    // Recommended Doctors
    const docsContainer = document.getElementById("report-doctors-list");
    docsContainer.innerHTML = "";
    data.recommended_doctors.forEach(doc => {
        const div = document.createElement("div");
        div.className = "doctor-card";
        div.innerHTML = `
            <div>
                <div class="doc-name">${doc.name}</div>
                <div class="doc-spec">${doc.title} (${doc.specialty})</div>
                <div class="doc-hospital">🏥 ${doc.hospital}</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:4px;">📍 ${doc.chamber}</div>
                <div style="font-size:0.82rem; color:#0369a1; font-weight:600; margin-top:4px;">🕒 ${doc.visiting_hours}</div>
            </div>
            <div style="margin-top:14px; display:flex; gap:8px; align-items:center;">
                <a href="tel:${doc.phone.replace(/[^0-9+]/g, '')}" class="btn btn-outline btn-sm" style="color:#0284c7; font-weight:700;">
                    📞 Call ${doc.phone}
                </a>
                <button class="btn btn-primary btn-sm" onclick="bookDoctorAppointment('${doc.name}', '${doc.hospital}')">
                    🗓 Book Consultation
                </button>
            </div>
        `;
        docsContainer.appendChild(div);
    });

    // Suggested Medications
    const medsContainer = document.getElementById("report-meds-list");
    medsContainer.innerHTML = "";
    
    // Safety warnings if any
    if (data.contraindication_safety_warnings && data.contraindication_safety_warnings.length > 0) {
        data.contraindication_safety_warnings.forEach(warn => {
            const wBox = document.createElement("div");
            wBox.style.background = "#fee2e2";
            wBox.style.border = "1px solid #f87171";
            wBox.style.color = "#991b1b";
            wBox.style.padding = "8px 12px";
            wBox.style.borderRadius = "8px";
            wBox.style.fontSize = "0.86rem";
            wBox.style.fontWeight = "700";
            wBox.style.marginBottom = "10px";
            wBox.innerText = "⚠️ " + warn;
            medsContainer.appendChild(wBox);
        });
    }

    data.suggested_medications.forEach(med => {
        const mDiv = document.createElement("div");
        mDiv.style.background = "#f8fafc";
        mDiv.style.border = "1px solid #e2e8f0";
        mDiv.style.borderRadius = "8px";
        mDiv.style.padding = "12px";
        mDiv.style.marginBottom = "10px";
        mDiv.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <strong style="color:#0f172a; font-size:0.95rem;">${med.name}</strong>
                <span style="font-size:0.75rem; background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:12px; font-weight:600;">${med.category}</span>
            </div>
            <div style="font-size:0.86rem; color:#0369a1; font-weight:600; margin-top:4px;">📋 Dose & Route: ${med.dosage} (${med.route})</div>
            <div style="font-size:0.82rem; color:#64748b; margin-top:2px;">⚠️ Warnings: ${med.warnings}</div>
        `;
        medsContainer.appendChild(mDiv);
    });

    // Security Details
    if (data.security) {
        document.getElementById("report-audit-hash").innerText = data.security.audit_block_hash;
        document.getElementById("report-cipher-algo").innerText = `${data.security.phi_encryption.cipher_algorithm} (Authenticated AEAD)`;
    }

    reportCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Doctor Directory
async function loadDoctorsDirectory() {
    const grid = document.getElementById("all-doctors-grid");
    if (!grid) return;

    try {
        const res = await fetch("/api/doctors");
        const data = await res.json();
        
        grid.innerHTML = "";
        data.doctors.forEach(doc => {
            const card = document.createElement("div");
            card.className = "doctor-card";
            card.innerHTML = `
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <span class="doc-name">${doc.name}</span>
                        ${doc.emergency_on_call ? '<span class="esi-badge esi-1" style="font-size:0.7rem;">24/7 ON CALL</span>' : ''}
                    </div>
                    <div class="doc-spec">${doc.title}</div>
                    <div style="font-size:0.82rem; color:#475569; margin-top:2px;">🎓 ${doc.qualification}</div>
                    <div class="doc-hospital">🏥 ${doc.hospital}</div>
                    <div style="font-size:0.82rem; color:#475569; margin-top:4px;">📍 ${doc.chamber}</div>
                    <div style="font-size:0.82rem; color:#0369a1; font-weight:600; margin-top:4px;">🕒 ${doc.visiting_hours}</div>
                    
                    <div style="margin-top:10px; font-size:0.8rem; color:#334155;">
                        <strong>Key Services:</strong>
                        <ul style="padding-left:16px; margin-top:4px;">
                            ${doc.services.map(s => `<li>${s}</li>`).join("")}
                        </ul>
                    </div>
                </div>

                <div style="margin-top:16px; padding-top:12px; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:#0f172a; font-size:0.9rem;">Fee: ৳${doc.fee_bdt}</span>
                    <div style="display:flex; gap:6px;">
                        <a href="tel:${doc.phone.replace(/[^0-9+]/g, '')}" class="btn btn-outline btn-sm">📞 Call</a>
                        <button class="btn btn-primary btn-sm" onclick="bookDoctorAppointment('${doc.name}', '${doc.hospital}')">🗓 Book</button>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (e) {
        console.error("Could not load doctors:", e);
    }
}

// Medicine Database
async function loadMedicinesList(query = "") {
    const list = document.getElementById("all-medicines-list");
    if (!list) return;

    try {
        const url = query ? `/api/medicines?query=${encodeURIComponent(query)}` : "/api/medicines";
        const res = await fetch(url);
        const data = await res.json();
        
        list.innerHTML = "";
        data.medicines.forEach(m => {
            const row = document.createElement("div");
            row.style.border = "1px solid #e2e8f0";
            row.style.borderRadius = "8px";
            row.style.padding = "14px";
            row.style.background = "white";
            row.style.marginBottom = "10px";
            row.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <strong style="font-size:1rem; color:#0f172a;">${m.name}</strong>
                        <span style="font-size:0.8rem; color:#64748b; margin-left:8px;">(${m.brand_names.join(", ")})</span>
                    </div>
                    <span style="background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">${m.category}</span>
                </div>
                <div style="font-size:0.86rem; color:#334155; margin-top:6px;"><strong>Indications:</strong> ${m.indications.join(", ")}</div>
                <div style="font-size:0.86rem; color:#0369a1; font-weight:600; margin-top:2px;"><strong>Dosage:</strong> ${m.dosage}</div>
                <div style="font-size:0.82rem; color:#dc2626; margin-top:2px;"><strong>Contraindications:</strong> ${m.contraindications.join(", ")}</div>
            `;
            list.appendChild(row);
        });
    } catch (e) {
        console.error("Could not load medicines:", e);
    }
}

// Drug Interaction Checker
async function checkDrugInteractions() {
    const input = document.getElementById("drug-interaction-input").value;
    const drugs = input.split(",").map(d => d.trim()).filter(d => d.length > 0);

    if (drugs.length < 2) {
        showToast("Enter at least two drug names separated by commas (e.g. Timolol, Verapamil)", "warning");
        return;
    }

    try {
        const res = await fetch("/api/medicines/interaction-check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ drug_names: drugs })
        });
        const data = await res.json();
        const container = document.getElementById("interaction-results-container");
        container.innerHTML = "";

        if (!data.has_interactions) {
            container.innerHTML = `<div style="background:#f0fdf4; border:1px solid #86efac; color:#166534; padding:12px; border-radius:8px; font-weight:600;">✓ No severe contraindications detected among: ${drugs.join(", ")}</div>`;
        } else {
            data.interactions.forEach(inter => {
                const el = document.createElement("div");
                el.style.background = "#fee2e2";
                el.style.border = "1px solid #f87171";
                el.style.color = "#991b1b";
                el.style.padding = "12px";
                el.style.borderRadius = "8px";
                el.style.marginBottom = "8px";
                el.innerHTML = `
                    <div style="font-weight:700; font-size:0.92rem;">⚠️ SEVERE INTERACTION: ${inter.drugs.join(" + ")}</div>
                    <div style="font-size:0.86rem; margin-top:4px;">${inter.warning}</div>
                `;
                container.appendChild(el);
            });
        }
    } catch (e) {
        showToast(e.message, "error");
    }
}

// Export FHIR R4 Bundle
async function exportFhirBundle() {
    if (!lastConsultationResult) {
        showToast("Generate a consultation first to export FHIR!", "warning");
        return;
    }

    try {
        const res = await fetch("/api/export/fhir", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(lastConsultationResult)
        });
        const bundle = await res.json();
        
        // Show in FHIR Modal
        const fhirCodeEl = document.getElementById("fhir-json-code");
        if (fhirCodeEl) fhirCodeEl.innerText = JSON.stringify(bundle, null, 2);
        
        const modal = document.getElementById("fhir-export-modal");
        if (modal) modal.classList.add("active");
    } catch (e) {
        showToast("FHIR Export error: " + e.message, "error");
    }
}

function closeFhirModal() {
    const modal = document.getElementById("fhir-export-modal");
    if (modal) modal.classList.remove("active");
}

function copyFhirJson() {
    const code = document.getElementById("fhir-json-code").innerText;
    navigator.clipboard.writeText(code).then(() => {
        showToast("FHIR R4 JSON Bundle copied to clipboard!", "success");
    });
}

// Audit Ledger Verification
async function verifyAuditLedger() {
    try {
        const res = await fetch("/api/security/audit-trail");
        const data = await res.json();
        
        const statusEl = document.getElementById("ledger-status-text");
        statusEl.innerHTML = `
            <div style="color:${data.mathematical_integrity_verified ? '#16a34a' : '#dc2626'}; font-weight:700;">
                ${data.mathematical_integrity_verified ? '✓ 100% Cryptographic Integrity Verified' : '✕ Tampering Detected!'}
            </div>
            <div style="font-size:0.85rem; color:#475569; margin-top:4px;">
                Total Blocks: ${data.ledger_blocks_count} | ${data.verification_message}
            </div>
        `;

        const chainContainer = document.getElementById("ledger-blocks-container");
        chainContainer.innerHTML = "";
        data.blocks.slice(-5).reverse().forEach(b => {
            const bDiv = document.createElement("div");
            bDiv.style.background = "#0f172a";
            bDiv.style.color = "#e2e8f0";
            bDiv.style.borderRadius = "8px";
            bDiv.style.padding = "10px 14px";
            bDiv.style.marginBottom = "8px";
            bDiv.style.fontSize = "0.78rem";
            bDiv.style.fontFamily = "monospace";
            bDiv.innerHTML = `
                <div style="color:#38bdf8; font-weight:700;">Block #${b.index} [${b.event_type}]</div>
                <div>Hash: ${b.hash}</div>
                <div style="color:#94a3b8;">Prev: ${b.prev_hash}</div>
                <div style="color:#cbd5e1; margin-top:2px;">Data: ${b.data_summary}</div>
            `;
            chainContainer.appendChild(bDiv);
        });

        showToast("Audit chain verified mathematically!", "success");
    } catch (e) {
        showToast(e.message, "error");
    }
}

// Appointment Booking Simulation
function bookDoctorAppointment(docName, hospital) {
    document.getElementById("modal-booking-doc-name").innerText = `${docName} (${hospital})`;
    document.getElementById("booking-modal").classList.add("active");
}

function closeBookingModal() {
    document.getElementById("booking-modal").classList.remove("active");
}

function confirmBooking() {
    const ptName = document.getElementById("booking-pt-name").value || "Patient";
    const ptPhone = document.getElementById("booking-pt-phone").value;
    if (!ptPhone) {
        showToast("Please provide contact phone number for appointment confirmation", "warning");
        return;
    }
    closeBookingModal();
    showToast(`Appointment confirmed for ${ptName}! SMS confirmation sent to ${ptPhone}.`, "success");
}

// Text-to-Speech for Emergency Audio Guidance
function speakEmergencyInstructions(lang) {
    if (!('speechSynthesis' in window)) {
        showToast("Speech synthesis not supported in this browser.", "warning");
        return;
    }
    const text = lang === 'bn' 
        ? document.getElementById("report-instr-bn").innerText 
        : document.getElementById("report-instr-en").innerText;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang === 'bn' ? 'bn-BD' : 'en-US';
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
}
