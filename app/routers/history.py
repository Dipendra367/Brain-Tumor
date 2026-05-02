from fastapi import APIRouter, HTTPException, Header
from firebase_admin import firestore
from firebase_config import db, PREDICTIONS_COLLECTION
from routers.auth import verify_token
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

router = APIRouter()

# ── Schema ────────────────────────────────────────────────
class SavePredictionRequest(BaseModel):
    report_id: str
    patient_name: str
    patient_age: Optional[str] = ""
    patient_gender: Optional[str] = ""
    prediction_class: str
    confidence: float
    all_scores: dict
    info: str
    severity: str
    gradcam_overlay: str    # base64 overlay image

# ── POST /api/history/save ────────────────────────────────
@router.post("/history/save")
def save_prediction(
    data: SavePredictionRequest,
    authorization: str = Header(...)
):
    """
    Doctor calls this after prediction to save to Firestore.
    Requires Bearer token.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    id_token = authorization.split(" ")[1]
    decoded  = verify_token(id_token)
    uid      = decoded["uid"]

    doc_data = {
        "report_id"        : data.report_id,
        "doctor_uid"       : uid,
        "patient_name"     : data.patient_name,
        "patient_age"      : data.patient_age,
        "patient_gender"   : data.patient_gender,
        "prediction_class" : data.prediction_class,
        "confidence"       : data.confidence,
        "all_scores"       : data.all_scores,
        "info"             : data.info,
        "severity"         : data.severity,
        "gradcam_overlay"  : data.gradcam_overlay,
        "created_at"       : firestore.SERVER_TIMESTAMP,
    }

    # Save with report_id as document ID so patient can look it up
    db.collection(PREDICTIONS_COLLECTION).document(data.report_id).set(doc_data)

    return {
        "message"   : "Prediction saved successfully",
        "report_id" : data.report_id,
    }

# ── GET /api/history ──────────────────────────────────────
@router.get("/history")
def get_history(authorization: str = Header(...)):
    """
    Returns all predictions for the logged-in doctor.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    id_token = authorization.split(" ")[1]
    decoded  = verify_token(id_token)
    uid      = decoded["uid"]

    docs = db.collection(PREDICTIONS_COLLECTION)\
             .where("doctor_uid", "==", uid)\
             .order_by("created_at", direction=firestore.Query.DESCENDING)\
             .stream()

    results = []
    for doc in docs:
        data = doc.to_dict()
        # Remove heavy base64 from list view
        data.pop("gradcam_overlay", None)
        results.append(data)

    return {"predictions": results, "count": len(results)}

# ── GET /api/history/{report_id} ──────────────────────────
@router.get("/history/{report_id}")
def get_prediction(report_id: str, authorization: str = Header(...)):
    """
    Returns a single prediction by report_id.
    Used by doctor to review old results + by patient viewer.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    doc = db.collection(PREDICTIONS_COLLECTION).document(report_id).get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    return doc.to_dict()

# ── GET /api/history/patient/{report_id} ──────────────────
@router.get("/history/patient/{report_id}")
def get_patient_result(report_id: str):
    """
    Public endpoint — no auth required.
    Patient enters report ID to view their diagnosis.
    """
    doc = db.collection(PREDICTIONS_COLLECTION).document(report_id).get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Report ID {report_id} not found")

    data = doc.to_dict()

    # Return only patient-safe fields (no doctor UID)
    return {
        "report_id"        : data.get("report_id"),
        "patient_name"     : data.get("patient_name"),
        "prediction_class" : data.get("prediction_class"),
        "confidence"       : data.get("confidence"),
        "all_scores"       : data.get("all_scores"),
        "info"             : data.get("info"),
        "severity"         : data.get("severity"),
        "gradcam_overlay"  : data.get("gradcam_overlay"),
        "created_at"       : data.get("created_at"),
    }