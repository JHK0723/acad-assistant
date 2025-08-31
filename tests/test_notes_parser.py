import pytest
import asyncio
from app.agents.notes_parser import notes_parser_agent
from app.models import NotesParseRequest


@pytest.mark.asyncio
async def test_notes_parser_basic():
    """Test basic notes parsing functionality"""
    request = NotesParseRequest(
        content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        extract_keywords=True,
        extract_concepts=True,
        extract_questions=False
    )
    
    result = await notes_parser_agent.parse_notes(request)
    
    assert result.success is True
    assert len(result.parsed_content) > 0
    assert isinstance(result.keywords, list)
    assert isinstance(result.concepts, list)


@pytest.mark.asyncio
async def test_notes_parser_with_questions():
    """Test notes parsing with question generation"""
    request = NotesParseRequest(
        content="Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes.",
        extract_keywords=True,
        extract_concepts=True,
        extract_questions=True
    )
    
    result = await notes_parser_agent.parse_notes(request)
    
    assert result.success is True
    assert len(result.parsed_content) > 0
    assert isinstance(result.study_questions, list)


def test_notes_parser_validation():
    """Test input validation for notes parser"""
    with pytest.raises(ValueError):
        NotesParseRequest(
            content="",  # Empty content should raise validation error
            extract_keywords=True
        )
