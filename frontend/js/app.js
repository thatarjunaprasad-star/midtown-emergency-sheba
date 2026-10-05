/**
 * Midtown Emergency Sheba - Main Application Script
 * Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
 */

// Tab Navigation
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSystemStatus();
    loadSampleChips();
    loadDoctorsDirectory();
    loadMedicinesList();
});

function initTabs() {
    const tabBtns = document.querySelectorAll(".nav-tab-btn");
    const panes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            
            tabBtns.forEach(b => b.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });
}

async function initSystemStatus() {
    try {
        const res = await fetch("/api/status");
        if (res.ok) {
            const data = await res.json();
            console.log("System Online:", data);
        }
    } catch (e) {
        console.warn("API offline or starting up:", e);
    }
}

function showToast(message, type = "info") {
    let container = document.getElementById("toast-container");
    if (!container) {
        container = document.createElement("div");
        container.id = "toast-container";
        container.style.position = "fixed";
        container.style.bottom = "20px";
        container.style.right = "20px";
        container.style.zIndex = "9999";
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "8px";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.style.padding = "10px 18px";
    toast.style.borderRadius = "8px";
    toast.style.fontSize = "0.88rem";
    toast.style.fontWeight = "600";
    toast.style.color = "white";
    toast.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
    toast.style.transition = "all 0.3s";
    
    if (type === "success") toast.style.background = "#16a34a";
    else if (type === "error") toast.style.background = "#dc2626";
    else if (type === "warning") toast.style.background = "#d97706";
    else toast.style.background = "#0284c7";

    toast.innerText = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// Emergency SOS Modal
function openEmergencyModal() {
    const modal = document.getElementById("emergency-sos-modal");
    if (modal) modal.classList.add("active");
}

function closeEmergencyModal() {
    const modal = document.getElementById("emergency-sos-modal");
    if (modal) modal.classList.remove("active");
}
