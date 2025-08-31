from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
import asyncio
import structlog
from app.models import NotesParseRequest, NotesParseResponse, ErrorResponse
from app.agents.notes_parser import notes_parser_agent
from app.utils.config import settings

logger = structlog.get_logger()
router = APIRouter(prefix="/notes", tags=["Notes Parser"])


@router.post("/parse", response_model=NotesParseResponse)
async def parse_notes(request: NotesParseRequest):
    """
    Parse educational notes and extract structured information
    
    - **content**: The text content to parse (required)
    - **file_format**: Format of the input content (pdf, docx, txt, md)
    - **extract_keywords**: Whether to extract keywords (default: True)
    - **extract_concepts**: Whether to extract concepts (default: True)
    - **extract_questions**: Whether to generate study questions (default: False)
    """
    try:
        # Validate content length
        if len(request.content) > settings.max_content_length:
            raise HTTPException(
                status_code=413,
                detail=f"Content too long. Maximum {settings.max_content_length} characters allowed."
            )
        
        logger.info("Processing notes parsing request", 
                   content_length=len(request.content),
                   file_format=request.file_format)
        
        # Process the request
        result = await notes_parser_agent.parse_notes(request)
        
        logger.info("Notes parsing completed", 
                   success=result.success,
                   processing_time=result.processing_time)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Notes parsing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/parse-file", response_model=NotesParseResponse)
async def parse_notes_from_file(
    file: UploadFile = File(...),
    extract_keywords: bool = True,
    extract_concepts: bool = True,
    extract_questions: bool = False
):
    """
    Parse educational notes from an uploaded file
    
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
        
        # Determine file format
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'txt'
        
        # Extract text content based on file type
        if file_extension == 'txt' or file_extension == 'md':
            text_content = content.decode('utf-8')
        elif file_extension == 'docx':
            # For DOCX files, you would use python-docx
            text_content = content.decode('utf-8', errors='ignore')  # Simplified for demo
        elif file_extension == 'pdf':
            # For PDF files, you would use PyPDF2 or similar
            text_content = content.decode('utf-8', errors='ignore')  # Simplified for demo
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create parse request
        request = NotesParseRequest(
            content=text_content,
            file_format=file_extension,
            extract_keywords=extract_keywords,
            extract_concepts=extract_concepts,
            extract_questions=extract_questions
        )
        
        logger.info("Processing file parsing request", 
                   filename=file.filename,
                   file_size=len(content),
                   file_format=file_extension)
        
        # Process the request
        result = await notes_parser_agent.parse_notes(request)
        
        logger.info("File parsing completed", 
                   success=result.success,
                   processing_time=result.processing_time)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parsing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def notes_parser_health():
    """Health check for the Notes Parser agent"""
    try:
        # Test a simple parsing operation
        test_request = NotesParseRequest(
            content="This is a test content for health check.",
            extract_keywords=False,
            extract_concepts=False,
            extract_questions=False
        )
        
        result = await notes_parser_agent.parse_notes(test_request)
        
        return {
            "status": "healthy" if result.success else "degraded",
            "agent": "NotesParserAgent",
            "primary_provider": notes_parser_agent.primary_provider,
            "test_successful": result.success
        }
        
    except Exception as e:
        logger.error("Notes parser health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "agent": "NotesParserAgent",
            "error": str(e)
        }
