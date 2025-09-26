from pydantic import BaseModel, Field, field_validator
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
    content: str = Field(..., min_length=1, max_length=50000)
    file_format: Optional[FileFormat] = FileFormat.TXT
    extract_keywords: bool = True
    extract_concepts: bool = True
    extract_questions: bool = False
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class SummarizeRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100000)
    summary_type: str = "comprehensive"
    max_length: int = Field(default=500, ge=50, le=2000)
    focus_areas: Optional[List[str]] = None
    
    @field_validator('summary_type')
    def validate_summary_type(cls, v):
        allowed = ["bullet_points", "comprehensive", "abstract", "key_points"]
        if v not in allowed:
            raise ValueError(f'Summary type must be one of: {", ".join(allowed)}')
        return v

class KeywordExtraction(BaseModel):
    keyword: str
    importance_score: float = Field(ge=0.0, le=1.0)
    context: Optional[str] = None

class ConceptExtraction(BaseModel):
    concept: str
    definition: Optional[str] = None
    related_terms: List[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)

class StudyQuestion(BaseModel):
    question: str
    difficulty_level: str
    question_type: str
    suggested_answer: Optional[str] = None

class NotesParseResponse(BaseModel):
    success: bool
    message: str
    parsed_content: str
    keywords: List[KeywordExtraction] = Field(default_factory=list)
    concepts: List[ConceptExtraction] = Field(default_factory=list)
    study_questions: List[StudyQuestion] = Field(default_factory=list)
    processing_time: float
    agent_used: str

class SummaryResponse(BaseModel):
    success: bool
    message: str
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_points: List[str] = Field(default_factory=list)
    processing_time: float
    agent_used: str

# Models for Question Paper Generation (NEW)
class QuestionPaperRequest(BaseModel):
    syllabus_text: str = Field(..., description="The syllabus content for generating the paper")
    test_type: str = Field(..., description="Type of test: CAT-1, CAT-2, or FAT")
    modules: List[str] = Field(..., min_items=1, description="List of modules to be covered")
    difficulty: str = Field("medium", description="User-selected difficulty (will be overridden by test_type logic)")
    title: str = "Question Paper"

class QuestionPart(BaseModel):
    label: Optional[str] = None
    marks: int
    text: str
    module: List[str]
    difficulty: str

class Question(BaseModel):
    q_no: int
    marks: int
    parts: List[QuestionPart]
    instructions: Optional[str] = None

class PaperMetadata(BaseModel):
    title: str
    test_type: str
    modules: List[str]
    difficulty: str
    total_marks: int
    notes: Optional[str] = None

class PaperValidation(BaseModel):
    total_marks_check: int
    unique_questions: bool

class QuestionPaperData(BaseModel):
    metadata: PaperMetadata
    paper: List[Question]
    validation: PaperValidation

class QuestionPaperResponse(BaseModel):
    success: bool
    message: str
    paper_data: Optional[QuestionPaperData]
    processing_time: float
    agent_used: str
# End of New Models

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    agents_status: Dict[str, str]

