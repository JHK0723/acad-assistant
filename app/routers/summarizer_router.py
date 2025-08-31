from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
import structlog
from app.models import SummarizeRequest, SummaryResponse, ErrorResponse
from app.agents.summarizer import summarizer_agent
from app.utils.config import settings

logger = structlog.get_logger()
router = APIRouter(prefix="/summarize", tags=["Summarizer"])


@router.post("/", response_model=SummaryResponse)
async def summarize_content(request: SummarizeRequest):
    """
    Summarize educational content
    
    - **content**: The text content to summarize (required)
    - **summary_type**: Type of summary - bullet_points, comprehensive, abstract, key_points
    - **max_length**: Maximum length of summary in words (50-2000)
    - **focus_areas**: Specific areas to focus on (optional)
    - **include_examples**: Whether to include examples in summary
    """
    try:
        # Validate content length
        if len(request.content) > settings.max_content_length:
            raise HTTPException(
                status_code=413,
                detail=f"Content too long. Maximum {settings.max_content_length} characters allowed."
            )
        
        logger.info("Processing summarization request", 
                   content_length=len(request.content),
                   summary_type=request.summary_type,
                   max_length=request.max_length)
        
        # Process the request
        result = await summarizer_agent.summarize_content(request)
        
        logger.info("Summarization completed", 
                   success=result.success,
                   processing_time=result.processing_time,
                   compression_ratio=result.compression_ratio)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Summarization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/file", response_model=SummaryResponse)
async def summarize_file(
    file: UploadFile = File(...),
    summary_type: str = "comprehensive",
    max_length: int = 500,
    focus_areas: Optional[List[str]] = None
):
    """
    Summarize content from an uploaded file
    
    Supported file formats: .txt, .md, .docx, .pdf
    """
    try:
        # Validate file size
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        
        # Read file content
        content = await file.read()
        if len(content) > max_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum {settings.max_file_size_mb}MB allowed."
            )
        
        # Determine file format and extract text
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'txt'
        
        if file_extension in ['txt', 'md']:
            text_content = content.decode('utf-8')
        elif file_extension == 'docx':
            # Simplified - in production, use python-docx
            text_content = content.decode('utf-8', errors='ignore')
        elif file_extension == 'pdf':
            # Simplified - in production, use PyPDF2
            text_content = content.decode('utf-8', errors='ignore')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create summarize request
        request = SummarizeRequest(
            content=text_content,
            summary_type=summary_type,
            max_length=max_length,
            focus_areas=focus_areas
        )
        
        logger.info("Processing file summarization request", 
                   filename=file.filename,
                   file_size=len(content),
                   summary_type=summary_type)
        
        # Process the request
        result = await summarizer_agent.summarize_content(request)
        
        logger.info("File summarization completed", 
                   success=result.success,
                   processing_time=result.processing_time)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File summarization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/bullet-points")
async def generate_bullet_summary(
    content: str,
    max_points: int = 8
):
    """Generate a bullet point summary"""
    try:
        if len(content) > settings.max_content_length:
            raise HTTPException(status_code=413, detail="Content too long")
        
        result = await summarizer_agent.generate_bullet_summary(content, max_points)
        
        return {
            "success": True,
            "bullet_points": result,
            "total_points": len(result)
        }
        
    except Exception as e:
        logger.error("Bullet summary generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/abstract")
async def generate_abstract(
    content: str,
    max_length: int = 200
):
    """Generate an academic abstract"""
    try:
        if len(content) > settings.max_content_length:
            raise HTTPException(status_code=413, detail="Content too long")
        
        result = await summarizer_agent.generate_abstract(content, max_length)
        
        return {
            "success": True,
            "abstract": result,
            "word_count": len(result.split())
        }
        
    except Exception as e:
        logger.error("Abstract generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def summarizer_health():
    """Health check for the Summarizer agent"""
    try:
        # Test a simple summarization operation
        test_request = SummarizeRequest(
            content="This is a test content for health check. It contains multiple sentences to test the summarization capability.",
            summary_type="comprehensive",
            max_length=50
        )
        
        result = await summarizer_agent.summarize_content(test_request)
        
        return {
            "status": "healthy" if result.success else "degraded",
            "agent": "SummarizerAgent",
            "primary_provider": summarizer_agent.primary_provider,
            "test_successful": result.success
        }
        
    except Exception as e:
        logger.error("Summarizer health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "agent": "SummarizerAgent",
            "error": str(e)
        }
