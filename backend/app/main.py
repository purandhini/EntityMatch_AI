"""Main FastAPI server entry point."""

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from backend.app.api.routes import router as api_router
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app():
    """Initializes and configures the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is not installed. Install requirements.txt to run the API service.")

    app = FastAPI(
        title="EntityMatch AI Service",
        description="REST API for Business Entity Resolution and Matching Interpretability",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")
    return app


if HAS_FASTAPI:
    app = create_app()
else:
    app = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
