from fastapi import APIRouter, HTTPException, Header
from firebase_admin import auth, firestore
from firebase_config import db, USERS_COLLECTION, PREDICTIONS_COLLECTION
from routers.auth import verify_token
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# ── Schema ────────────────────────────────────────────────
class ToggleDoctorRequest(BaseModel):
    uid: str
    active: bool

# ── Helper: verify admin role ─────────────────────────────
def require_admin(authorization: str):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    id_token = authorization.split(" ")[1]
    decoded  = verify_token(id_token)
    uid      = decoded["uid"]

    doc = db.collection(USERS_COLLECTION).document(uid).get()
    if not doc.exists or doc.to_dict().get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return uid

# ── GET /api/admin/stats ──────────────────────────────────
@router.get("/admin/stats")
def get_stats(authorization: str = Header(...)):
    """System-wide diagnostic statistics for admin dashboard."""
    require_admin(authorization)

    # Count predictions per class
    predictions = db.collection(PREDICTIONS_COLLECTION).stream()
    stats = {
        "total_predictions" : 0,
        "by_class"          : {"Glioma": 0, "Meningioma": 0, "No Tumor": 0, "Pituitary": 0},
        "total_doctors"     : 0,
        "active_doctors"    : 0,
    }

    for doc in predictions:
        data = doc.to_dict()
        stats["total_predictions"] += 1
        cls = data.get("prediction_class", "")
        if cls in stats["by_class"]:
            stats["by_class"][cls] += 1

    # Count doctors
    doctors = db.collection(USERS_COLLECTION).where("role", "==", "doctor").stream()
    for doc in doctors:
        stats["total_doctors"] += 1
        if doc.to_dict().get("active", True):
            stats["active_doctors"] += 1

    return stats

# ── GET /api/admin/doctors ────────────────────────────────
@router.get("/admin/doctors")
def get_doctors(authorization: str = Header(...)):
    """Returns all doctor accounts."""
    require_admin(authorization)

    docs = db.collection(USERS_COLLECTION)\
             .where("role", "==", "doctor")\
             .stream()

    doctors = []
    for doc in docs:
        data = doc.to_dict()
        doctors.append({
            "uid"       : data.get("uid"),
            "full_name" : data.get("full_name"),
            "email"     : data.get("email"),
            "hospital"  : data.get("hospital"),
            "active"    : data.get("active", True),
        })

    return {"doctors": doctors, "count": len(doctors)}

# ── PATCH /api/admin/doctors/toggle ──────────────────────
@router.patch("/admin/doctors/toggle")
def toggle_doctor(data: ToggleDoctorRequest, authorization: str = Header(...)):
    """Enable or disable a doctor account."""
    require_admin(authorization)

    doc_ref = db.collection(USERS_COLLECTION).document(data.uid)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Doctor not found")

    doc_ref.update({"active": data.active})

    status = "enabled" if data.active else "disabled"
    return {"message": f"Doctor account {status} successfully", "uid": data.uid}