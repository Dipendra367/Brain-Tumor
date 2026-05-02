from fastapi import APIRouter, UploadFile, File, HTTPException
from predictor import predict_from_bytes
from gradcam import generate_gradcam
import uuid

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_SIZE_MB   = 10


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a brain MRI image → get prediction + Grad-CAM heatmap.

    Returns:
        - report_id      : unique ID for this prediction
        - prediction     : class, confidence, all_scores, info, severity
        - gradcam        : base64 original, heatmap, overlay images
    """
    # ── Validate file type ────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Upload a JPEG or PNG MRI image."
        )

    # ── Read bytes ────────────────────────────────────────
    img_bytes = await file.read()

    # ── Validate file size ────────────────────────────────
    if len(img_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_SIZE_MB}MB."
        )

    # ── Run prediction ────────────────────────────────────
    try:
        prediction = predict_from_bytes(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # ── Generate Grad-CAM ─────────────────────────────────
    try:
        gradcam = generate_gradcam(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")

    # ── Generate report ID ────────────────────────────────
    report_id = f"BD-2026-{str(uuid.uuid4())[:4].upper()}"

    return {
        "report_id"  : report_id,
        "prediction" : prediction,
        "gradcam"    : gradcam,
    }