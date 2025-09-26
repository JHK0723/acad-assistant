import time
import re
from typing import Dict, Any, List
import structlog
from app.utils.openai_client import openai_client
from app.utils.ollama_client import ollama_client
from app.models import SummarizeRequest, SummaryResponse

logger = structlog.get_logger()


class SummarizerAgent:
    """Agent responsible for summarizing educational content"""
    
    def __init__(self, primary_provider: str = "ollama"):
        self.primary_provider = primary_provider
        self.fallback_provider = "openai" if primary_provider == "ollama" else "ollama"
    
    async def summarize_content(self, request: SummarizeRequest) -> SummaryResponse:
        """Main method to summarize content using the configured provider"""
        start_time = time.time()
        original_length = len(request.content.split())
        
        try:
            # Try primary provider first
            result = await self._summarize_with_provider(request, self.primary_provider)
            agent_used = self.primary_provider
            
        except Exception as e:
            logger.warning(f"Primary provider {self.primary_provider} failed", error=str(e))
            try:
                # Fallback to secondary provider
                result = await self._summarize_with_provider(request, self.fallback_provider)
                agent_used = f"{self.fallback_provider} (fallback)"
            except Exception as fallback_error:
                logger.error(f"Both providers failed", 
                           primary_error=str(e), fallback_error=str(fallback_error))
                return SummaryResponse(
                    success=False,
                    message=f"Both AI providers failed: {str(e)}",
                    summary="",
                    original_length=original_length,
                    summary_length=0,
                    compression_ratio=0.0,
                    processing_time=time.time() - start_time,
                    agent_used="none"
                )
        
        processing_time = time.time() - start_time
        summary = result.get("summary", "")
        summary_length = len(summary.split())
        compression_ratio = summary_length / original_length if original_length > 0 else 0.0
        
        # Extract key points from the summary
        key_points = self._extract_key_points(summary, request.summary_type)
        
        return SummaryResponse(
            success=True,
            message="Content summarized successfully",
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            key_points=key_points,
            processing_time=processing_time,
            agent_used=agent_used
        )
    
    async def _summarize_with_provider(self, request: SummarizeRequest, provider: str) -> Dict[str, Any]:
        """Summarize content with a specific provider"""
        if provider == "ollama":
            return await ollama_client.summarize_with_ollama(
                content=request.content,
                summary_type=request.summary_type,
                max_length=request.max_length,
                focus_areas=request.focus_areas
            )
        elif provider == "openai":
            return await openai_client.summarize_with_openai(
                content=request.content,
                summary_type=request.summary_type,
                max_length=request.max_length,
                focus_areas=request.focus_areas
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _extract_key_points(self, summary: str, summary_type: str) -> List[str]:
        """Extract key points from the summary text"""
        key_points = []
        
        if summary_type == "bullet_points":
            # Look for bullet points or numbered lists
            bullet_patterns = [
                r'[•\-\*]\s*(.+?)(?=\n[•\-\*]|\n\n|$)',
                r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)',
                r'[-•]\s*(.+?)(?=\n[-•]|\n\n|$)'
            ]
            
            for pattern in bullet_patterns:
                matches = re.findall(pattern, summary, re.MULTILINE)
                if matches:
                    key_points.extend([match.strip() for match in matches[:8]])
                    break
        
        else:
            # For comprehensive or abstract summaries, extract sentences
            sentences = re.split(r'[.!?]+', summary)
            # Take every other sentence or the longest sentences
            sorted_sentences = sorted(sentences, key=len, reverse=True)
            key_points = [s.strip() for s in sorted_sentences[:6] if s.strip() and len(s.strip()) > 20]
        
        return key_points[:8]  # Return max 8 key points
    
    async def generate_bullet_summary(self, content: str, max_points: int = 8) -> List[str]:
        """Generate a bullet point summary specifically"""
        request = SummarizeRequest(
            content=content,
            summary_type="bullet_points",
            max_length=max_points * 25  # Approximate words per bullet point
        )
        
        result = await self.summarize_content(request)
        return result.key_points
    
    async def generate_abstract(self, content: str, max_length: int = 200) -> str:
        """Generate an academic abstract"""
        request = SummarizeRequest(
            content=content,
            summary_type="abstract",
            max_length=max_length
        )
        
        result = await self.summarize_content(request)
        return result.summary
    
    async def generate_comprehensive_summary(self, content: str, focus_areas: List[str] = None) -> str:
        """Generate a comprehensive summary with optional focus areas"""
        request = SummarizeRequest(
            content=content,
            summary_type="comprehensive",
            max_length=600,
            focus_areas=focus_areas
        )
        
        result = await self.summarize_content(request)
        return result.summary


# Global summarizer agent instance
summarizer_agent = SummarizerAgent(primary_provider="ollama")
