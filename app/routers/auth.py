from fastapi import APIRouter, HTTPException, Header
from firebase_admin import auth, firestore
from firebase_config import db, USERS_COLLECTION
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str          # "admin" or "doctor"
    full_name: str
    hospital: Optional[str] = ""

class LoginRequest(BaseModel):
    id_token: str      # Firebase ID token from frontend after login

# ── Helper: verify Firebase token ─────────────────────────
def verify_token(id_token: str) -> dict:
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

def get_user_role(uid: str) -> str:
    doc = db.collection(USERS_COLLECTION).document(uid).get()
    if doc.exists:
        return doc.to_dict().get("role", "unknown")
    raise HTTPException(status_code=404, detail="User not found in Firestore")

# ── POST /api/auth/register ───────────────────────────────
@router.post("/auth/register")
def register(data: RegisterRequest):
    """
    Creates a Firebase Auth user and stores role in Firestore.
    Only admin can create doctor accounts (enforced on frontend).
    """
    if data.role not in ["admin", "doctor"]:
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'doctor'")

    try:
        # Create user in Firebase Auth
        user = auth.create_user(
            email=data.email,
            password=data.password,
            display_name=data.full_name,
        )

        # Store role + profile in Firestore
        db.collection(USERS_COLLECTION).document(user.uid).set({
            "uid"       : user.uid,
            "email"     : data.email,
            "full_name" : data.full_name,
            "role"      : data.role,
            "hospital"  : data.hospital,
            "created_at": firestore.SERVER_TIMESTAMP,
            "active"    : True,
        })

        return {
            "message"   : f"{data.role.capitalize()} account created successfully",
            "uid"       : user.uid,
            "email"     : data.email,
            "role"      : data.role,
        }

    except auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── POST /api/auth/login ──────────────────────────────────
@router.post("/auth/login")
def login(data: LoginRequest):
    """
    Verifies Firebase ID token sent from frontend after login.
    Returns user profile + role from Firestore.
    """
    decoded = verify_token(data.id_token)
    uid = decoded["uid"]

    doc = db.collection(USERS_COLLECTION).document(uid).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User profile not found")

    user_data = doc.to_dict()

    if not user_data.get("active", True):
        raise HTTPException(status_code=403, detail="Account disabled by admin")

    return {
        "uid"       : uid,
        "email"     : user_data.get("email"),
        "full_name" : user_data.get("full_name"),
        "role"      : user_data.get("role"),
        "hospital"  : user_data.get("hospital"),
    }

# ── GET /api/auth/me ──────────────────────────────────────
@router.get("/auth/me")
def me(authorization: str = Header(...)):
    """
    Returns current user profile. Expects 'Bearer <id_token>' header.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    id_token = authorization.split(" ")[1]
    decoded = verify_token(id_token)
    uid = decoded["uid"]

    doc = db.collection(USERS_COLLECTION).document(uid).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    return doc.to_dict()