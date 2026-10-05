/**
 * Sequential Vitals & Trajectory Recurrent Neural Network (RNN / Bi-LSTM) Tracker
 * Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
 */

let lastRnnResult = null;

let currentVitalsData = [
    { time: "08:00", hr: 76, sbp: 122, dbp: 80, spo2: 98, temp: 36.8, rr: 16, pain: 2 },
    { time: "10:00", hr: 88, sbp: 128, dbp: 82, spo2: 97, temp: 37.2, rr: 18, pain: 4 },
    { time: "12:00", hr: 112, sbp: 148, dbp: 92, spo2: 94, temp: 38.4, rr: 23, pain: 7 }
];

document.addEventListener("DOMContentLoaded", () => {
    renderVitalsTable();
    // Run initial inference
    runRnnAnalysis();
});

function renderVitalsTable() {
    const tbody = document.getElementById("vitals-table-body");
    if (!tbody) return;

    tbody.innerHTML = "";
    currentVitalsData.forEach((row, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><input type="text" class="form-control" style="padding:4px 8px; width:80px;" value="${row.time}" onchange="updateVitalRow(${idx}, 'time', this.value)"></td>
            <td><input type="number" class="form-control" style="padding:4px 8px; width:70px;" value="${row.hr}" onchange="updateVitalRow(${idx}, 'hr', Number(this.value))"></td>
            <td><input type="number" class="form-control" style="padding:4px 8px; width:70px;" value="${row.sbp}" onchange="updateVitalRow(${idx}, 'sbp', Number(this.value))"></td>
            <td><input type="number" class="form-control" style="padding:4px 8px; width:70px;" value="${row.dbp}" onchange="updateVitalRow(${idx}, 'dbp', Number(this.value))"></td>
            <td><input type="number" class="form-control" style="padding:4px 8px; width:70px;" value="${row.spo2}" onchange="updateVitalRow(${idx}, 'spo2', Number(this.value))"></td>
            <td><input type="number" step="0.1" class="form-control" style="padding:4px 8px; width:70px;" value="${row.temp}" onchange="updateVitalRow(${idx}, 'temp', Number(this.value))"></td>
            <td><input type="number" class="form-control" style="padding:4px 8px; width:65px;" value="${row.rr}" onchange="updateVitalRow(${idx}, 'rr', Number(this.value))"></td>
            <td><input type="number" min="0" max="10" class="form-control" style="padding:4px 8px; width:65px;" value="${row.pain}" onchange="updateVitalRow(${idx}, 'pain', Number(this.value))"></td>
            <td><button class="btn btn-outline btn-sm" onclick="removeVitalRow(${idx})" style="color:#dc2626;">✕</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function updateVitalRow(idx, field, value) {
    currentVitalsData[idx][field] = value;
}

function addVitalRow() {
    const last = currentVitalsData[currentVitalsData.length - 1] || { hr: 75, sbp: 120, dbp: 80, spo2: 98, temp: 37, rr: 16, pain: 0 };
    const nextHour = `${(currentVitalsData.length * 2 + 8) % 24}:00`.padStart(5, '0');
    currentVitalsData.push({
        time: nextHour,
        hr: last.hr,
        sbp: last.sbp,
        dbp: last.dbp,
        spo2: last.spo2,
        temp: last.temp,
        rr: last.rr,
        pain: last.pain
    });
    renderVitalsTable();
}

function removeVitalRow(idx) {
    if (currentVitalsData.length <= 1) {
        showToast("At least one reading is required for trajectory evaluation", "warning");
        return;
    }
    currentVitalsData.splice(idx, 1);
    renderVitalsTable();
}

function loadVitalsPreset(presetName) {
    if (presetName === "shock") {
        currentVitalsData = [
            { time: "08:00", hr: 82, sbp: 120, dbp: 78, spo2: 98, temp: 37.0, rr: 16, pain: 3 },
            { time: "10:00", hr: 104, sbp: 102, dbp: 68, spo2: 95, temp: 38.5, rr: 22, pain: 6 },
            { time: "12:00", hr: 128, sbp: 86, dbp: 54, spo2: 91, temp: 39.2, rr: 28, pain: 8 }
        ];
    } else if (presetName === "hypertensive") {
        currentVitalsData = [
            { time: "08:00", hr: 78, sbp: 145, dbp: 95, spo2: 98, temp: 36.8, rr: 16, pain: 4 },
            { time: "10:00", hr: 84, sbp: 172, dbp: 108, spo2: 97, temp: 37.0, rr: 18, pain: 6 },
            { time: "12:00", hr: 92, sbp: 195, dbp: 122, spo2: 96, temp: 37.1, rr: 20, pain: 9 }
        ];
    } else {
        // Stable
        currentVitalsData = [
            { time: "08:00", hr: 72, sbp: 118, dbp: 76, spo2: 99, temp: 36.6, rr: 15, pain: 1 },
            { time: "10:00", hr: 74, sbp: 120, dbp: 78, spo2: 98, temp: 36.8, rr: 16, pain: 2 },
            { time: "12:00", hr: 75, sbp: 122, dbp: 80, spo2: 98, temp: 36.7, rr: 16, pain: 1 }
        ];
    }
    renderVitalsTable();
    runRnnAnalysis();
}

async function runRnnAnalysis() {
    try {
        const res = await fetch("/api/rnn/analyze-vitals", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_id: "PT-MIDTOWN-01",
                readings: currentVitalsData
            })
        });
        if (!res.ok) throw new Error("RNN analysis error: " + res.statusText);
        const data = await res.json();
        renderRnnResults(data);
    } catch (e) {
        showToast(e.message, "error");
    }
}

function renderRnnResults(data) {
    lastRnnResult = data;
    
    // Risk score gauge
    const gauge = document.getElementById("rnn-risk-gauge");
    const statusText = document.getElementById("rnn-trajectory-status");
    const escalationText = document.getElementById("rnn-escalation");
    const shockText = document.getElementById("rnn-shock-index");
    const mapText = document.getElementById("rnn-map");

    if (gauge) {
        gauge.innerText = `${data.deterioration_risk_percent}%`;
        gauge.className = "gauge-circle " + (
            data.status_color === "red" ? "gauge-red" :
            data.status_color === "amber" ? "gauge-amber" : "gauge-green"
        );
    }

    if (statusText) statusText.innerText = data.trajectory_status;
    if (escalationText) escalationText.innerText = data.escalation_recommendation;
    if (shockText) shockText.innerText = `${data.shock_index} (${data.shock_index > 0.9 ? 'Shock Alert' : 'Normal'})`;
    if (mapText) mapText.innerText = `${data.mean_arterial_pressure} mmHg`;

    // Clinical Alerts List
    const alertsContainer = document.getElementById("rnn-alerts-list");
    if (alertsContainer) {
        alertsContainer.innerHTML = "";
        if (data.active_clinical_alerts.length === 0) {
            alertsContainer.innerHTML = `<div style="color:#16a34a; font-weight:600;">✓ Physiological parameters within stable limits.</div>`;
        } else {
            data.active_clinical_alerts.forEach(alert => {
                const el = document.createElement("div");
                el.style.background = "#fee2e2";
                el.style.color = "#991b1b";
                el.style.padding = "6px 12px";
                el.style.borderRadius = "6px";
                el.style.fontSize = "0.85rem";
                el.style.fontWeight = "600";
                el.style.marginBottom = "6px";
                el.innerText = alert;
                alertsContainer.appendChild(el);
            });
        }
    }

    // Render SVG Trend Graph
    renderTrendSvg(data.time_series_metrics);

    // Render Attention Weights
    renderAttentionWeights(data.time_series_metrics, data.temporal_attention_weights);
}

function renderTrendSvg(metrics) {
    const svg = document.getElementById("rnn-trend-svg");
    if (!svg || !metrics || metrics.length === 0) return;

    const width = 500;
    const height = 180;
    const padding = 35;

    const xStep = (width - padding * 2) / Math.max(1, metrics.length - 1);
    
    // Extract points
    const hrPoints = metrics.map((m, i) => [padding + i * xStep, height - padding - ((m.hr - 40) / 140) * (height - padding * 2)]);
    const sbpPoints = metrics.map((m, i) => [padding + i * xStep, height - padding - ((m.sbp - 60) / 160) * (height - padding * 2)]);
    const spo2Points = metrics.map((m, i) => [padding + i * xStep, height - padding - ((m.spo2 - 70) / 30) * (height - padding * 2)]);

    function toPath(points) {
        return points.map((p, i) => (i === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`)).join(" ");
    }

    svg.innerHTML = `
        <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#cbd5e1" stroke-width="2"/>
        <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#cbd5e1" stroke-width="2"/>
        
        <!-- HR Curve (Red) -->
        <path d="${toPath(hrPoints)}" fill="none" stroke="#ef4444" stroke-width="3" stroke-linecap="round"/>
        <!-- SBP Curve (Blue) -->
        <path d="${toPath(sbpPoints)}" fill="none" stroke="#0284c7" stroke-width="3" stroke-linecap="round"/>
        <!-- SpO2 Curve (Green) -->
        <path d="${toPath(spo2Points)}" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round"/>

        <!-- Dots & Labels -->
        ${metrics.map((m, i) => `
            <circle cx="${padding + i * xStep}" cy="${hrPoints[i][1]}" r="4" fill="#ef4444"/>
            <circle cx="${padding + i * xStep}" cy="${sbpPoints[i][1]}" r="4" fill="#0284c7"/>
            <text x="${padding + i * xStep}" y="${height - 12}" font-size="11" text-anchor="middle" fill="#64748b">${m.time}</text>
        `).join("")}
    `;
}

function renderAttentionWeights(metrics, weights) {
    const container = document.getElementById("rnn-attention-container");
    if (!container || !weights) return;

    container.innerHTML = "";
    metrics.forEach((m, idx) => {
        const w = weights[idx] || 0;
        const pct = Math.round(w * 100);
        const div = document.createElement("div");
        div.style.marginBottom = "8px";
        div.innerHTML = `
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; font-weight:600; margin-bottom:2px;">
                <span>Step ${idx+1} (${m.time})</span>
                <span>Attention Weight: ${pct}%</span>
            </div>
            <div style="height:6px; background:#e2e8f0; border-radius:3px; overflow:hidden;">
                <div style="width:${pct}%; height:100%; background:${pct > 40 ? '#ef4444' : '#0284c7'}; border-radius:3px;"></div>
            </div>
        `;
        container.appendChild(div);
    });
}

function transferRnnToConsultation() {
    if (!lastRnnResult) {
        showToast("Run RNN vitals analysis first!", "warning");
        return;
    }
    const consultTabBtn = document.querySelector('[data-tab="tab-consultation"]');
    if (consultTabBtn) consultTabBtn.click();
    showToast("RNN Vitals trajectory transferred to consultation engine!", "success");
}
