import ollama
import asyncio
import httpx
from typing import Dict, Any, Optional
import structlog
from app.utils.config import settings

logger = structlog.get_logger()


class OllamaClient:
    """Ollama client wrapper for local Mistral 7B model"""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
    
    async def generate_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate completion using Ollama API"""
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if system_message:
                full_prompt = f"System: {system_message}\n\nUser: {prompt}"
            
            # Use httpx for async HTTP requests to Ollama
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error("Ollama API error", error=str(e))
            raise Exception(f"Ollama API error: {str(e)}")
    
    async def parse_notes_with_ollama(self, content: str, extract_keywords: bool = True,
                                    extract_concepts: bool = True, extract_questions: bool = False) -> Dict[str, Any]:
        """Parse notes using Ollama Mistral 7B"""
        system_message = """You are an expert academic assistant specialized in parsing and analyzing educational content.
        Analyze the provided content and extract structured information in a clear, organized format."""
        
        prompt_parts = [
            f"Analyze this educational content and provide structured analysis:",
            f"\nContent: {content[:3000]}...",  # Limit content for token efficiency
            "\nProvide analysis including:"
        ]
        
        if extract_keywords:
            prompt_parts.append("1. Important keywords with relevance scores")
        
        if extract_concepts:
            prompt_parts.append("2. Key concepts with definitions")
        
        if extract_questions:
            prompt_parts.append("3. Study questions for review")
        
        prompt_parts.append("\nFormat your response clearly and concisely.")
        
        prompt = "\n".join(prompt_parts)
        
        response = await self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,
            max_tokens=1200
        )
        
        return {"parsed_content": response}
    
    async def summarize_with_ollama(self, content: str, summary_type: str = "comprehensive",
                                  max_length: int = 500, focus_areas: Optional[list] = None) -> Dict[str, Any]:
        """Summarize content using Ollama Mistral 7B"""
        system_message = f"""You are an expert at creating {summary_type} summaries. 
        Create a clear, concise summary in approximately {max_length} words."""
        
        prompt_parts = [
            f"Create a {summary_type} summary of this content:",
            f"\nContent: {content[:4000]}...",  # Limit for token efficiency
            f"\nRequirements:",
            f"- Style: {summary_type}",
            f"- Target length: {max_length} words"
        ]
        
        if focus_areas:
            prompt_parts.append(f"- Focus areas: {', '.join(focus_areas)}")
        
        prompt_parts.append("\nProvide a well-structured summary:")
        
        prompt = "\n".join(prompt_parts)
        
        response = await self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,
            max_tokens=max_length + 100
        )
        
        return {"summary": response}
    
    async def health_check(self) -> bool:
        """Check if Ollama server is running and model is available"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # Check if our model is available
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                return any(self.model in name for name in model_names)
                
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return False


# Global Ollama client instance
ollama_client = OllamaClient()
