import firebase_admin
from firebase_admin import credentials, firestore
import os

# ── Config ────────────────────────────────────────────────
# firebase-credentials.json sits in project root (one level above app/)
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRED_PATH = os.path.join(BASE_DIR, "firebase-credentials.json")

# ── Initialize Firebase (only once) ──────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate(CRED_PATH)
    firebase_admin.initialize_app(cred)
    print(f"✅ Firebase initialized: braindetect-1842e")

# ── Firestore client ──────────────────────────────────────
db = firestore.client()

# ── Collection names ──────────────────────────────────────
USERS_COLLECTION       = "users"
PREDICTIONS_COLLECTION = "predictions"
CHATS_COLLECTION       = "chats"