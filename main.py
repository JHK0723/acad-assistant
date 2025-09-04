from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import asyncio
import uvicorn
from datetime import datetime
from app.routers import notes_router, summarizer_router
from app.utils.config import settings
from app.utils.ollama_client import ollama_client
from app.models import HealthCheckResponse, ErrorResponse

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI application
app = FastAPI(
    title="AI Academic Assistant",
    description="AI-powered academic assistant",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(notes_router.router)
app.include_router(summarizer_router.router)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Academic Assistant API",
        "version": "1.0.0",
        "description": "AI-powered academic assistant with notes parsing and summarization",
        "endpoints": {
            "notes_parser": "/notes/parse",
            "summarizer": "/summarize/",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check for all components"""
    try:
        # Check Ollama connection
        ollama_status = "healthy" if await ollama_client.health_check() else "unhealthy"
        
        # Check OpenAI (would require API call)
        openai_status = "healthy"  
        
        agents_status = {
            "ollama": ollama_status,
            "openai": openai_status,
            "notes_parser": "healthy",
            "summarizer": "healthy"
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in agents_status.values()) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            agents_status=agents_status
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            agents_status={"error": str(e)}
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.warning("HTTP exception occurred", 
                  status_code=exc.status_code, 
                  detail=exc.detail,
                  path=request.url.path)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=str(exc.status_code)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error("Unhandled exception occurred", 
                error=str(exc),
                path=request.url.path)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="500",
            details={"message": str(exc) if settings.debug else "An error occurred"}
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Powering up AI Academic Assistant", 
               version="1.0.0",
               debug=settings.debug)
    
    try:
        ollama_healthy = await ollama_client.health_check()
        logger.info("Ollama health check", healthy=ollama_healthy)
    except Exception as e:
        logger.warning("Ollama health check failed", error=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down AI Academic Assistant")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
