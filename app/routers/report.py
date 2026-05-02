from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import Response
from firebase_config import db, PREDICTIONS_COLLECTION, USERS_COLLECTION
from routers.auth import verify_token
from pdf_generator import generate_pdf_report

router = APIRouter()

# ── GET /api/report/{report_id} ───────────────────────────
# Doctor endpoint — requires auth token
@router.get("/report/{report_id}")
def download_report(report_id: str, authorization: str = Header(...)):
    """
    Generates and returns a PDF diagnostic report.
    Requires doctor Bearer token.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    id_token = authorization.split(" ")[1]
    decoded  = verify_token(id_token)
    uid      = decoded["uid"]

    # Get prediction from Firestore
    pred_doc = db.collection(PREDICTIONS_COLLECTION).document(report_id).get()
    if not pred_doc.exists:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    pred = pred_doc.to_dict()

    # Get doctor info
    doctor_doc = db.collection(USERS_COLLECTION).document(uid).get()
    doctor     = doctor_doc.to_dict() if doctor_doc.exists else {}

    pdf_bytes = _generate(pred, doctor)

    return Response(
        content    = pdf_bytes,
        media_type = "application/pdf",
        headers    = {
            "Content-Disposition": f'attachment; filename="BrainDetect_{report_id}.pdf"'
        }
    )

# ── GET /api/report/patient/{report_id} ───────────────────
# Patient endpoint — NO auth required, just report ID
@router.get("/report/patient/{report_id}")
def download_patient_report(report_id: str):
    """
    Public endpoint — patient downloads their own PDF using report ID only.
    No authentication required.
    """
    pred_doc = db.collection(PREDICTIONS_COLLECTION).document(report_id).get()
    if not pred_doc.exists:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    pred = pred_doc.to_dict()

    # Get doctor info for the report
    doctor_uid = pred.get("doctor_uid", "")
    doctor     = {}
    if doctor_uid:
        doctor_doc = db.collection(USERS_COLLECTION).document(doctor_uid).get()
        if doctor_doc.exists:
            doctor = doctor_doc.to_dict()

    pdf_bytes = _generate(pred, doctor)

    return Response(
        content    = pdf_bytes,
        media_type = "application/pdf",
        headers    = {
            "Content-Disposition": f'attachment; filename="BrainDetect_{report_id}.pdf"'
        }
    )

# ── Shared PDF generator ───────────────────────────────────
def _generate(pred: dict, doctor: dict) -> bytes:
    try:
        return generate_pdf_report(
            report_id           = pred.get("report_id", ""),
            patient_name        = pred.get("patient_name", ""),
            patient_age         = pred.get("patient_age", ""),
            patient_gender      = pred.get("patient_gender", ""),
            doctor_name         = doctor.get("full_name", ""),
            hospital            = doctor.get("hospital", ""),
            prediction_class    = pred.get("prediction_class", ""),
            confidence          = pred.get("confidence", 0),
            all_scores          = pred.get("all_scores", {}),
            severity            = pred.get("severity", ""),
            gradcam_overlay_b64 = pred.get("gradcam_overlay", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")