"""
main.py
-------
ScienceQ FastAPI service — a thin HTTP transport over the LangGraph agent.

The sys.path bridge to the agent/ and pipeline/ packages is set up in
api/__init__.py, so it is already in place by the time this module's imports
run.

Run locally:
    uvicorn api.main:app --reload --port 8080

In the container the port comes from Cloud Run's $PORT (see Dockerfile).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .limiter import limiter
from .routes import router

app = FastAPI(title="ScienceQ API", version="1.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://scienceq-web-886463515307.europe-west1.run.app",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    """Liveness probe for Cloud Run."""
    return {"status": "ok"}
