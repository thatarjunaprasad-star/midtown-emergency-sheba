/**
 * Convolutional Vision AI (CNN) & Grad-CAM Viewer
 * Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
 */

let lastCnnResult = null;

async function loadSampleChips() {
    const container = document.getElementById("sample-chips-container");
    if (!container) return;

    try {
        const res = await fetch("/api/samples");
        const data = await res.json();
        
        container.innerHTML = "";
        data.samples.forEach(s => {
            const btn = document.createElement("button");
            btn.className = "sample-chip";
            btn.innerText = "🔍 " + s.title;
            btn.onclick = () => runCnnWithSample(s.sample_id);
            container.appendChild(btn);
        });
    } catch (e) {
        console.error("Could not load samples:", e);
    }
}

async function runCnnWithSample(sampleId) {
    showCnnLoading(true);
    const formData = new FormData();
    formData.append("sample_id", sampleId);

    try {
        const res = await fetch("/api/cnn/analyze", {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error("CNN Analysis failed: " + res.statusText);
        const report = await res.json();
        renderCnnResults(report);
        showToast("CNN inference completed with Grad-CAM heatmap!", "success");
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        showCnnLoading(false);
    }
}

async function handleImageUpload(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];

    showCnnLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/cnn/analyze", {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error("CNN inference failed: " + res.statusText);
        const report = await res.json();
        renderCnnResults(report);
        showToast("CNN inference & Grad-CAM analysis complete!", "success");
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        showCnnLoading(false);
    }
}

function showCnnLoading(isLoading) {
    const loader = document.getElementById("cnn-loading-indicator");
    const results = document.getElementById("cnn-results-card");
    if (loader) loader.style.display = isLoading ? "block" : "none";
    if (results && isLoading) results.style.opacity = "0.4";
}

function renderCnnResults(report) {
    lastCnnResult = report;
    const resultsCard = document.getElementById("cnn-results-card");
    if (!resultsCard) return;

    resultsCard.style.opacity = "1";
    resultsCard.style.display = "block";

    // Diagnostic Heading & Badges
    document.getElementById("cnn-pred-title").innerText = report.title;
    document.getElementById("cnn-confidence").innerText = `${report.confidence_percent}% Match`;
    
    const severityBadge = document.getElementById("cnn-severity-badge");
    severityBadge.innerText = report.severity;
    severityBadge.className = "esi-badge " + (
        report.severity.includes("CRITICAL") ? "esi-1" :
        report.severity === "HIGH" ? "esi-2" :
        report.severity === "MODERATE" ? "esi-3" : "esi-5"
    );

    document.getElementById("cnn-urgency").innerText = report.urgency;
    document.getElementById("cnn-findings").innerText = report.clinical_findings;
    document.getElementById("cnn-recommendations").innerText = report.recommendation;

    // Grad-CAM Visuals
    const visuals = report.visual_explanation;
    document.getElementById("cnn-img-original").src = visuals.original_image;
    document.getElementById("cnn-img-overlay").src = visuals.gradcam_overlay;
    document.getElementById("cnn-img-heatmap").src = visuals.heatmap_only;

    // Differentials list
    const diffContainer = document.getElementById("cnn-differentials-list");
    diffContainer.innerHTML = "";
    report.differentials.forEach(d => {
        const item = document.createElement("div");
        item.style.marginBottom = "8px";
        item.innerHTML = `
            <div style="display:flex; justify-content:space-between; font-size:0.84rem; font-weight:600; margin-bottom:2px;">
                <span>${d.title}</span>
                <span>${d.probability}%</span>
            </div>
            <div style="height:6px; background:#e2e8f0; border-radius:3px; overflow:hidden;">
                <div style="width:${d.probability}%; height:100%; background:var(--primary); border-radius:3px;"></div>
            </div>
        `;
        diffContainer.appendChild(item);
    });

    // Auto-scroll to results
    resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function transferCnnToConsultation() {
    if (!lastCnnResult) {
        showToast("Run a CNN scan first!", "warning");
        return;
    }
    // Switch to consultation tab
    const consultTabBtn = document.querySelector('[data-tab="tab-consultation"]');
    if (consultTabBtn) consultTabBtn.click();

    // Fill in chief complaint if empty
    const complaintInput = document.getElementById("consult-complaint");
    if (complaintInput && !complaintInput.value) {
        complaintInput.value = `Ocular evaluation requested. CNN Scan detected: ${lastCnnResult.title}. ${lastCnnResult.clinical_findings}`;
    }
    showToast("CNN diagnostic results transferred to consultation engine!", "success");
}
