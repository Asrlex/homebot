from fastapi import FastAPI
from prometheus_client import make_asgi_app
from app.api.routes import router as api_router

app = FastAPI(title="HomeBot ML Service", version="0.1.0")

# Include API routes
app.include_router(api_router, prefix="/api")

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
