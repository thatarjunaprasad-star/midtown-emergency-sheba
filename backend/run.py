"""
Server startup script for Midtown Emergency Sheba.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    print("=" * 70)
    print("MIDTOWN EMERGENCY SHEBA - AI Medical & Emergency Agent")
    print("Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.")
    print("Architectures: PyTorch Residual CNN (Grad-CAM), Bi-LSTM RNN, Multi-Agent LLM")
    print("Security: AES-256-GCM PHI, SHA-256 Audit Trail, HL7 FHIR R4 Standard")
    print("=" * 70)
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    uvicorn.run("backend.api:app", host=host, port=port, reload=False, log_level="info")
