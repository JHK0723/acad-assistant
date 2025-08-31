from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum
import re


class AgentType(str, Enum):
    NOTES_PARSER = "notes_parser"
    SUMMARIZER = "summarizer"


class FileFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class NotesParseRequest(BaseModel):
    """Request model for parsing notes"""
    content: str = Field(..., min_length=1, max_length=50000, description="Content to parse")
    file_format: Optional[FileFormat] = Field(default=FileFormat.TXT, description="Format of the input content")
    extract_keywords: bool = Field(default=True, description="Whether to extract keywords")
    extract_concepts: bool = Field(default=True, description="Whether to extract concepts")
    extract_questions: bool = Field(default=False, description="Whether to generate study questions")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or only whitespace')
        return v.strip()


class SummarizeRequest(BaseModel):
    """Request model for summarizing content"""
    content: str = Field(..., min_length=1, max_length=100000, description="Content to summarize")
    summary_type: str = Field(default="comprehensive", description="Type of summary: bullet_points, comprehensive, abstract")
    max_length: int = Field(default=500, ge=50, le=2000, description="Maximum length of summary in words")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific areas to focus on")
    include_examples: bool = Field(default=False, description="Whether to include examples in summary")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or only whitespace')
        return v.strip()
    
    @validator('summary_type')
    def validate_summary_type(cls, v):
        allowed_types = ["bullet_points", "comprehensive", "abstract", "key_points"]
        if v not in allowed_types:
            raise ValueError(f'Summary type must be one of: {", ".join(allowed_types)}')
        return v


class KeywordExtraction(BaseModel):
    """Model for extracted keywords"""
    keyword: str
    importance_score: float = Field(ge=0.0, le=1.0)
    context: Optional[str] = None


class ConceptExtraction(BaseModel):
    """Model for extracted concepts"""
    concept: str
    definition: Optional[str] = None
    related_terms: List[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)


class StudyQuestion(BaseModel):
    """Model for generated study questions"""
    question: str
    difficulty_level: str = Field(description="easy, medium, hard")
    question_type: str = Field(description="multiple_choice, short_answer, essay")
    suggested_answer: Optional[str] = None


class NotesParseResponse(BaseModel):
    """Response model for parsed notes"""
    success: bool
    message: str
    parsed_content: str
    keywords: List[KeywordExtraction] = Field(default_factory=list)
    concepts: List[ConceptExtraction] = Field(default_factory=list)
    study_questions: List[StudyQuestion] = Field(default_factory=list)
    processing_time: float
    agent_used: str


class SummaryResponse(BaseModel):
    """Response model for summarized content"""
    success: bool
    message: str
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_points: List[str] = Field(default_factory=list)
    processing_time: float
    agent_used: str


class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    agents_status: Dict[str, str]
