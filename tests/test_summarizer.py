import pytest
import asyncio
from app.agents.summarizer import summarizer_agent
from app.models import SummarizeRequest


@pytest.mark.asyncio
async def test_summarizer_basic():
    """Test basic summarization functionality"""
    request = SummarizeRequest(
        content="Artificial intelligence is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. AI has various subfields including machine learning, natural language processing, computer vision, and robotics.",
        summary_type="comprehensive",
        max_length=100
    )
    
    result = await summarizer_agent.summarize_content(request)
    
    assert result.success is True
    assert len(result.summary) > 0
    assert result.original_length > 0
    assert result.summary_length > 0
    assert result.compression_ratio > 0


@pytest.mark.asyncio
async def test_summarizer_bullet_points():
    """Test bullet point summarization"""
    request = SummarizeRequest(
        content="Python is a high-level programming language. It is known for its simplicity and readability. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It has a large standard library and active community.",
        summary_type="bullet_points",
        max_length=80
    )
    
    result = await summarizer_agent.summarize_content(request)
    
    assert result.success is True
    assert len(result.summary) > 0
    assert isinstance(result.key_points, list)


def test_summarizer_validation():
    """Test input validation for summarizer"""
    with pytest.raises(ValueError):
        SummarizeRequest(
            content="",  # Empty content should raise validation error
            summary_type="comprehensive"
        )
