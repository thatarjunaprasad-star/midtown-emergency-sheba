"""
Security and Cryptographic Engine for Midtown Emergency Sheba.
Implements:
1. AES-256-GCM authenticated encryption for Protected Health Information (PHI).
2. Merkle / Hash-chaining SHA-256 tamper-evident clinical audit ledger.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import os
import time
import json
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Generate or load a consistent 256-bit AES master key
MASTER_KEY_HEX = os.getenv("MIDTOWN_SECURITY_KEY")
if not MASTER_KEY_HEX:
    # 32 bytes = 256 bits
    _RAW_KEY = hashlib.sha256(b"Midtown-Emergency-Sheba-Comilla-Payal-Rani-Das-2026").digest()
else:
    _RAW_KEY = bytes.fromhex(MASTER_KEY_HEX)

aes_gcm = AESGCM(_RAW_KEY)

def encrypt_phi(data_dict_or_str):
    """
    Encrypts sensitive medical data using AES-256-GCM.
    Returns nonce and ciphertext in Base64 encoding.
    """
    if isinstance(data_dict_or_str, (dict, list)):
        plaintext = json.dumps(data_dict_or_str).encode("utf-8")
    else:
        plaintext = str(data_dict_or_str).encode("utf-8")
        
    nonce = os.urandom(12)  # 96-bit nonce for AES-GCM standard
    ciphertext = aes_gcm.encrypt(nonce, plaintext, None)
    
    return {
        "cipher_algorithm": "AES-256-GCM",
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
        "encrypted_at": time.time()
    }


def decrypt_phi(encrypted_obj):
    """
    Decrypts AES-256-GCM ciphertext back into Python object or string.
    """
    nonce = base64.b64decode(encrypted_obj["nonce"])
    ciphertext = base64.b64decode(encrypted_obj["ciphertext"])
    plaintext = aes_gcm.decrypt(nonce, ciphertext, None)
    
    try:
        return json.loads(plaintext.decode("utf-8"))
    except Exception:
        return plaintext.decode("utf-8")


class AuditLedger:
    """
    Cryptographic Tamper-Evident Ledger for Clinical Consultations & Prescriptions.
    Each event is hashed with the previous block's hash, forming an immutable chain.
    """
    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = {
            "index": 0,
            "timestamp": time.time(),
            "event_type": "GENESIS_SECURITY_ANCHOR",
            "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
            "prev_hash": "0" * 64,
            "data_summary": "Midtown Emergency Sheba Secure Ledger Initialized"
        }
        genesis["hash"] = self._compute_hash(genesis)
        self.chain.append(genesis)

    def _compute_hash(self, block):
        block_string = f"{block['index']}|{block['timestamp']}|{block['event_type']}|{block['prev_hash']}|{block['data_summary']}"
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    def record_event(self, event_type, data_summary):
        prev_block = self.chain[-1]
        new_block = {
            "index": len(self.chain),
            "timestamp": time.time(),
            "event_type": event_type,
            "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
            "prev_hash": prev_block["hash"],
            "data_summary": str(data_summary)
        }
        new_block["hash"] = self._compute_hash(new_block)
        self.chain.append(new_block)
        return new_block

    def verify_integrity(self):
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr["prev_hash"] != prev["hash"]:
                return False, f"Broken link at block #{i}: prev_hash does not match #{i-1} hash!"
            if self._compute_hash(curr) != curr["hash"]:
                return False, f"Tampered block detected at #{i}: recorded hash does not match computed hash!"
        return True, "All ledger blocks intact. Zero tampering detected."

# Global singleton audit ledger
GLOBAL_AUDIT_LEDGER = AuditLedger()
