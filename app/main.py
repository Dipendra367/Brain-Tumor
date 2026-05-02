from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import predict, auth, history, report, admin
from predictor import load_model_once


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup — load model once into memory ──────────────
    print("🚀 BrainDetect API starting up...")
    load_model_once()
    print("✅ Model loaded and ready")
    yield
    # ── Shutdown ───────────────────────────────────────────
    print("🛑 BrainDetect API shutting down...")


app = FastAPI(
    title="BrainDetect API",
    description="AI-powered brain tumour detection with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow frontend (Vercel) and local dev ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────
app.include_router(predict.router,  prefix="/api")
app.include_router(auth.router,     prefix="/api")
app.include_router(history.router,  prefix="/api")
app.include_router(report.router,   prefix="/api")
app.include_router(admin.router,    prefix="/api")


@app.get("/")
def root():
    return {
        "message": "BrainDetect API is running",
        "docs":    "/docs",
        "health":  "/api/health",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "loaded"}