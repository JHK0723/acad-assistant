import openai
from typing import Dict, Any, Optional
import asyncio
import structlog
from app.utils.config import settings

logger = structlog.get_logger()


class OpenAIClient:
    """OpenAI client wrapper for the application"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    async def generate_completion(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        """Generate completion using OpenAI API"""
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def parse_notes_with_openai(self, content: str, extract_keywords: bool = True, 
                                    extract_concepts: bool = True, extract_questions: bool = False) -> Dict[str, Any]:
        """Parse notes using OpenAI"""
        system_message = """You are an expert academic assistant specialized in parsing and analyzing educational content.
        Your task is to analyze the provided content and extract structured information."""
        
        prompt_parts = [
            f"Please analyze the following content and provide a structured response:",
            f"\nContent: {content}",
            "\nPlease provide the following analysis:"
        ]
        
        if extract_keywords:
            prompt_parts.append("1. Extract 5-10 important keywords with importance scores (0.0-1.0)")
        
        if extract_concepts:
            prompt_parts.append("2. Identify key concepts with definitions and related terms")
        
        if extract_questions:
            prompt_parts.append("3. Generate 3-5 study questions with different difficulty levels")
        
        prompt_parts.append("\nFormat your response as a clear, structured analysis.")
        
        prompt = "\n".join(prompt_parts)
        
        response = await self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            max_tokens=1500,
            temperature=0.3
        )
        
        return {"parsed_content": response}
    
    async def summarize_with_openai(self, content: str, summary_type: str = "comprehensive",
                                  max_length: int = 500, focus_areas: Optional[list] = None) -> Dict[str, Any]:
        """Summarize content using OpenAI"""
        system_message = f"""You are an expert summarization assistant. Create a {summary_type} summary 
        of the provided content in approximately {max_length} words."""
        
        prompt_parts = [
            f"Please create a {summary_type} summary of the following content:",
            f"\nContent: {content}",
            f"\nSummary requirements:",
            f"- Type: {summary_type}",
            f"- Maximum length: {max_length} words",
        ]
        
        if focus_areas:
            prompt_parts.append(f"- Focus on these areas: {', '.join(focus_areas)}")
        
        prompt_parts.append("\nProvide a clear, concise summary that captures the main points.")
        
        prompt = "\n".join(prompt_parts)
        
        response = await self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            max_tokens=max_length + 200,
            temperature=0.3
        )
        
        return {"summary": response}


# Global OpenAI client instance
openai_client = OpenAIClient()
