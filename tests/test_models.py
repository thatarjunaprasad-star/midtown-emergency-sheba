"""
Unit tests for CNN, RNN, and LLM Agent in Midtown Emergency Sheba.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import os
import sys
from PIL import Image

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.cnn_module import run_cnn_inference, get_cnn_model
from backend.models.rnn_module import run_rnn_inference, get_rnn_model
from backend.models.llm_agent import run_llm_agent_consultation, match_doctors, find_safe_medicines

def test_cnn_initialization_and_inference():
    model, grad_cam = get_cnn_model()
    assert model is not None
    assert grad_cam is not None

    # Test with sample image
    sample_path = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "sample_cases", "corneal_ulcer.jpg")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            data = f.read()
        res = run_cnn_inference(data)
        assert "predicted_class" in res
        assert "confidence_percent" in res
        assert "visual_explanation" in res
        assert res["visual_explanation"]["gradcam_overlay"].startswith("data:image/jpeg;base64,")
        assert res["creator"] == "Payal Rani Das, Midtown Eye Hospital, Comilla"
        print(f"CNN Test Passed: Detected {res['predicted_class']} ({res['confidence_percent']}%)")

def test_rnn_vitals_trajectory():
    model = get_rnn_model()
    assert model is not None

    test_readings = [
        {"time": "08:00", "hr": 76, "sbp": 120, "dbp": 80, "spo2": 99, "temp": 36.8, "rr": 16, "pain": 2},
        {"time": "10:00", "hr": 88, "sbp": 130, "dbp": 85, "spo2": 97, "temp": 37.3, "rr": 18, "pain": 5},
        {"time": "12:00", "hr": 115, "sbp": 155, "dbp": 98, "spo2": 94, "temp": 38.6, "rr": 24, "pain": 8}
    ]
    res = run_rnn_inference(test_readings)
    assert "deterioration_risk_percent" in res
    assert "trajectory_status" in res
    assert "shock_index" in res
    assert len(res["time_series_metrics"]) == 3
    assert res["creator"] == "Payal Rani Das, Midtown Eye Hospital, Comilla"
    print(f"RNN Test Passed: Risk = {res['deterioration_risk_percent']}%, Status = {res['trajectory_status']}")

def test_llm_agent_consultation():
    sample_cnn = {
        "predicted_class": "corneal_ulcer_keratitis",
        "title": "Corneal Ulcer / Severe Microbial Keratitis",
        "confidence_percent": 96.5,
        "urgency": "Immediate Ophthalmic Emergency"
    }
    sample_rnn = {
        "deterioration_risk_percent": 45.0,
        "trajectory_status": "WORSENING TRAJECTORY",
        "shock_index": 0.74,
        "mean_arterial_pressure": 93.3,
        "latest_vitals": {"hr": 88, "sbp": 130, "dbp": 85, "spo2": 97, "temp": 37.3, "pain": 5}
    }

    consult = run_llm_agent_consultation(
        patient_name="Sultan Mahmud",
        age=52,
        gender="Male",
        chief_complaint="Severe right eye pain, redness, sensitivity to light after farm dust entered eye yesterday",
        medical_history="Mild asthma, hypertension",
        cnn_result=sample_cnn,
        rnn_result=sample_rnn
    )

    assert consult["creator"] == "Payal Rani Das, Midtown Eye Hospital, Comilla"
    assert consult["triage"]["esi_level"] in [1, 2]
    assert len(consult["recommended_doctors"]) > 0
    # First recommended doctor should be from Midtown Eye Hospital
    assert "Midtown Eye Hospital" in consult["recommended_doctors"][0]["hospital"]
    assert len(consult["suggested_medications"]) > 0
    assert "english" in consult["emergency_instructions"]
    assert "bengali" in consult["emergency_instructions"]
    print("LLM Agent Consultation Test Passed!")

if __name__ == "__main__":
    test_cnn_initialization_and_inference()
    test_rnn_vitals_trajectory()
    test_llm_agent_consultation()
    print("ALL MODEL TESTS PASSED SUCCESSFULLY!")
