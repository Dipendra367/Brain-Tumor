import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# ── Initialize Firebase (only once) ──────────────────────
# Credentials are loaded from the FIREBASE_CREDENTIALS environment variable.
# Set this in the Railway dashboard with the full JSON contents of your
# firebase-credentials.json file.
if not firebase_admin._apps:
    creds_json = os.environ.get("FIREBASE_CREDENTIALS")
    if not creds_json:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS environment variable is not set. "
            "Set it in the Railway dashboard with the contents of your "
            "firebase-credentials.json file."
        )
    cred_dict = json.loads(creds_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase initialized")

# ── Firestore client ──────────────────────────────────────
db = firestore.client()

# ── Collection names ──────────────────────────────────────
USERS_COLLECTION       = "users"
PREDICTIONS_COLLECTION = "predictions"
CHATS_COLLECTION       = "chats"
