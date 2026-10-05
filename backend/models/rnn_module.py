"""
Sequential Vitals & Trajectory Recurrent Neural Network (RNN / Bi-LSTM) Module
for Midtown Emergency Sheba.

Monitors time-series vital signs, computes shock index, temporal attention,
and forecasts clinical acute deterioration risk.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs shape: [batch_size, seq_len, hidden_dim]
        energy = self.attn(rnn_outputs)  # [batch_size, seq_len, 1]
        weights = F.softmax(energy, dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(weights * rnn_outputs, dim=1)  # [batch_size, hidden_dim]
        return context, weights


class ClinicalVitalsRNN(nn.Module):
    """
    High-level Bidirectional LSTM with Temporal Attention
    for multi-variate clinical time-series vitals & deterioration triage.
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM processes forward in time and contextually backward
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        bi_dim = hidden_dim * 2
        self.attention = TemporalAttention(bi_dim)
        
        # Risk prediction heads
        self.fc_risk = nn.Sequential(
            nn.Linear(bi_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Trajectory trend classifier (0: Improving, 1: Stable, 2: Deteriorating, 3: Critical)
        self.fc_trend = nn.Sequential(
            nn.Linear(bi_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, 128]
        context, attn_weights = self.attention(lstm_out)  # [batch_size, 128], [batch_size, seq_len, 1]
        
        deterioration_prob = self.fc_risk(context)
        trend_logits = self.fc_trend(context)
        return deterioration_prob, trend_logits, attn_weights


# Standard clinical normal references for normalization
NORM_REF = {
    "hr": {"mean": 75.0, "std": 15.0, "min": 35.0, "max": 180.0},
    "sbp": {"mean": 120.0, "std": 18.0, "min": 60.0, "max": 240.0},
    "dbp": {"mean": 80.0, "std": 12.0, "min": 40.0, "max": 140.0},
    "spo2": {"mean": 98.0, "std": 3.0, "min": 65.0, "max": 100.0},
    "temp": {"mean": 37.0, "std": 0.8, "min": 34.0, "max": 42.0},
    "rr": {"mean": 16.0, "std": 4.0, "min": 8.0, "max": 45.0},
    "pain": {"mean": 3.0, "std": 2.5, "min": 0.0, "max": 10.0}
}

_GLOBAL_RNN = None

def get_rnn_model():
    global _GLOBAL_RNN
    if _GLOBAL_RNN is None:
        _GLOBAL_RNN = ClinicalVitalsRNN(input_dim=7, hidden_dim=64, num_layers=2)
        _GLOBAL_RNN.eval()
    return _GLOBAL_RNN


def run_rnn_inference(time_series_vitals):
    """
    Inference entrypoint for Sequential Vitals & Trajectory:
    Accepts list of dictionaries or list of records representing sequential measurements:
    [
        {"time": "08:00", "hr": 78, "sbp": 122, "dbp": 80, "spo2": 98, "temp": 36.8, "rr": 16, "pain": 2},
        {"time": "10:00", "hr": 84, "sbp": 128, "dbp": 82, "spo2": 97, "temp": 37.1, "rr": 18, "pain": 4},
        {"time": "12:00", "hr": 102, "sbp": 145, "dbp": 94, "spo2": 95, "temp": 38.2, "rr": 22, "pain": 7}
    ]
    """
    model = get_rnn_model()
    
    if not time_series_vitals or len(time_series_vitals) == 0:
        raise ValueError("At least one vital signs reading is required")

    # If only 1 reading provided, generate prior baseline to form sequence
    if len(time_series_vitals) == 1:
        first = time_series_vitals[0]
        # create baseline synthetic reference
        baseline = {
            "time": "Baseline (-2h)",
            "hr": max(60, first.get("hr", 75) - 8),
            "sbp": max(100, first.get("sbp", 120) - 10),
            "dbp": max(70, first.get("dbp", 80) - 6),
            "spo2": min(100, first.get("spo2", 98) + 1),
            "temp": 36.9,
            "rr": max(14, first.get("rr", 16) - 2),
            "pain": max(0, first.get("pain", 2) - 2)
        }
        time_series_vitals = [baseline, first]

    # Convert to feature tensor
    seq_data = []
    clinical_metrics = []
    
    for row in time_series_vitals:
        hr = float(row.get("hr", 75))
        sbp = float(row.get("sbp", 120))
        dbp = float(row.get("dbp", 80))
        spo2 = float(row.get("spo2", 98))
        temp = float(row.get("temp", 37.0))
        rr = float(row.get("rr", 16))
        pain = float(row.get("pain", 0))
        
        # Clinical Derived Parameters
        map_val = (2 * dbp + sbp) / 3.0
        shock_index = hr / (sbp + 1e-5)
        pulse_press = sbp - dbp
        
        clinical_metrics.append({
            "time": row.get("time", "T"),
            "hr": hr,
            "sbp": sbp,
            "dbp": dbp,
            "spo2": spo2,
            "temp": temp,
            "rr": rr,
            "pain": pain,
            "map": round(map_val, 1),
            "shock_index": round(shock_index, 2),
            "pulse_pressure": round(pulse_press, 1)
        })

        # Z-score normalization for RNN
        feat = [
            (hr - NORM_REF["hr"]["mean"]) / NORM_REF["hr"]["std"],
            (sbp - NORM_REF["sbp"]["mean"]) / NORM_REF["sbp"]["std"],
            (dbp - NORM_REF["dbp"]["mean"]) / NORM_REF["dbp"]["std"],
            (spo2 - NORM_REF["spo2"]["mean"]) / NORM_REF["spo2"]["std"],
            (temp - NORM_REF["temp"]["mean"]) / NORM_REF["temp"]["std"],
            (rr - NORM_REF["rr"]["mean"]) / NORM_REF["rr"]["std"],
            (pain - NORM_REF["pain"]["mean"]) / NORM_REF["pain"]["std"]
        ]
        seq_data.append(feat)

    input_tensor = torch.tensor([seq_data], dtype=torch.float32)

    with torch.no_grad():
        det_prob_raw, trend_logits, attn_weights = model(input_tensor)

    # Latest values for acute risk calculation
    latest = clinical_metrics[-1]
    
    # Calculate rule-based clinical severity modifier (NEWS2 / qSOFA inspired)
    severity_score = 0
    alerts = []
    
    if latest["spo2"] < 92:
        severity_score += 3
        alerts.append("CRITICAL: Severe Hypoxemia (SpO2 < 92%). High risk of organ ischemia.")
    elif latest["spo2"] < 95:
        severity_score += 1
        alerts.append("WARNING: Mild Hypoxemia (SpO2 92-94%).")

    if latest["hr"] > 110:
        severity_score += 2
        alerts.append(f"Tachycardia alert: Heart rate {latest['hr']} bpm.")
    elif latest["hr"] < 50:
        severity_score += 2
        alerts.append(f"Bradycardia alert: Heart rate {latest['hr']} bpm. Timolol / beta-blocker contraindication!")

    if latest["sbp"] >= 180 or latest["dbp"] >= 120:
        severity_score += 3
        alerts.append("HYPERTENSIVE CRISIS: SBP >= 180 or DBP >= 120 mmHg. Risk of hypertensive retinopathy / intracranial event.")
    elif latest["sbp"] < 90:
        severity_score += 3
        alerts.append("HYPOTENSION ALERT: Systolic BP < 90 mmHg. Inadequate ocular and vital organ perfusion.")

    if latest["shock_index"] > 0.9:
        severity_score += 3
        alerts.append(f"HEMODYNAMIC SHOCK ALERT: Shock Index = {latest['shock_index']} (> 0.9). Decompensation imminent.")

    if latest["temp"] >= 39.0:
        severity_score += 2
        alerts.append(f"Hyperpyrexia: Temperature {latest['temp']} °C. Infectious etiology screening needed.")

    if latest["pain"] >= 8:
        severity_score += 1
        alerts.append("Severe agonizing pain reported (Score >= 8/10). Typical in acute angle glaucoma or corneal trauma.")

    # Blend RNN neural prediction with clinical rule scores
    base_rnn_prob = float(det_prob_raw.item())
    calibrated_prob = np.clip(base_rnn_prob * 0.4 + (severity_score / 12.0) * 0.6, 0.02, 0.99)
    
    # Trajectory classification
    if calibrated_prob > 0.70:
        trajectory_status = "CRITICAL DETERIORATION"
        status_color = "red"
        escalation_tier = "Immediate ICU / Resuscitation Bay Escalation"
    elif calibrated_prob > 0.40:
        trajectory_status = "WORSENING TRAJECTORY"
        status_color = "amber"
        escalation_tier = "Emergency Observation & Urgent Physician Review (Midtown Emergency Unit)"
    elif calibrated_prob > 0.18:
        trajectory_status = "STABLE BORDERLINE"
        status_color = "yellow"
        escalation_tier = "Standard Ward / Outpatient Monitoring"
    else:
        trajectory_status = "STABLE / IMPROVING"
        status_color = "green"
        escalation_tier = "Routine Ambulatory Sheba & Tele-monitoring"

    # Extract temporal attention distribution across time steps
    attn_list = [round(float(w), 3) for w in attn_weights[0, :, 0].numpy()]

    return {
        "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
        "deterioration_risk_percent": round(calibrated_prob * 100, 1),
        "trajectory_status": trajectory_status,
        "status_color": status_color,
        "escalation_recommendation": escalation_tier,
        "shock_index": latest["shock_index"],
        "mean_arterial_pressure": latest["map"],
        "latest_vitals": latest,
        "time_series_metrics": clinical_metrics,
        "temporal_attention_weights": attn_list,
        "active_clinical_alerts": alerts
    }
