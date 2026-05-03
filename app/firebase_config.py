import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# ── Load credentials ──────────────────────────────────────
# On Railway: credentials are in FIREBASE_CREDENTIALS env variable
# Locally: credentials are in firebase-credentials.json file

if not firebase_admin._apps:
    firebase_creds_env = os.environ.get("FIREBASE_CREDENTIALS")

    if firebase_creds_env:
        # Railway deployment — load from environment variable
        cred_dict = json.loads(firebase_creds_env)
        cred = credentials.Certificate(cred_dict)
        print("✅ Firebase initialized from environment variable")
    else:
        # Local development — load from file
        BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        CRED_PATH = os.path.join(BASE_DIR, "firebase-credentials.json")
        cred = credentials.Certificate(CRED_PATH)
        print(f"✅ Firebase initialized from file: braindetect-1842e")

    firebase_admin.initialize_app(cred)

# ── Firestore client ──────────────────────────────────────
db = firestore.client()

# ── Collection names ──────────────────────────────────────
USERS_COLLECTION       = "users"
PREDICTIONS_COLLECTION = "predictions"
CHATS_COLLECTION       = "chats"