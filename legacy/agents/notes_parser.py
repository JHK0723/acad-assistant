import time
import re
from typing import Dict, Any, List
import structlog
from app.utils.openai_client import openai_client
from app.utils.ollama_client import ollama_client
from app.models import (
    NotesParseRequest, NotesParseResponse, KeywordExtraction, 
    ConceptExtraction, StudyQuestion
)

logger = structlog.get_logger()


class NotesParserAgent:
    """Agent responsible for parsing and analyzing educational notes"""
    
    def __init__(self, primary_provider: str = "ollama"):
        self.primary_provider = primary_provider
        self.fallback_provider = "openai" if primary_provider == "ollama" else "ollama"
    
    async def parse_notes(self, request: NotesParseRequest) -> NotesParseResponse:
        """Main method to parse notes using the configured provider"""
        start_time = time.time()
        
        try:
            # Try primary provider first
            result = await self._parse_with_provider(request, self.primary_provider)
            agent_used = self.primary_provider
            
        except Exception as e:
            logger.warning(f"Primary provider {self.primary_provider} failed", error=str(e))
            try:
                # Fallback to secondary provider
                result = await self._parse_with_provider(request, self.fallback_provider)
                agent_used = f"{self.fallback_provider} (fallback)"
            except Exception as fallback_error:
                logger.error(f"Both providers failed", 
                           primary_error=str(e), fallback_error=str(fallback_error))
                return NotesParseResponse(
                    success=False,
                    message=f"Both AI providers failed: {str(e)}",
                    parsed_content="",
                    processing_time=time.time() - start_time,
                    agent_used="none"
                )
        
        processing_time = time.time() - start_time
        
        # Process the raw result into structured format
        parsed_response = await self._structure_response(result, request)
        
        return NotesParseResponse(
            success=True,
            message="Notes parsed successfully",
            parsed_content=parsed_response["parsed_content"],
            keywords=parsed_response.get("keywords", []),
            concepts=parsed_response.get("concepts", []),
            study_questions=parsed_response.get("study_questions", []),
            processing_time=processing_time,
            agent_used=agent_used
        )
    
    async def _parse_with_provider(self, request: NotesParseRequest, provider: str) -> Dict[str, Any]:
        """Parse notes with a specific provider"""
        if provider == "ollama":
            return await ollama_client.parse_notes_with_ollama(
                content=request.content,
                extract_keywords=request.extract_keywords,
                extract_concepts=request.extract_concepts,
                extract_questions=request.extract_questions
            )
        elif provider == "openai":
            return await openai_client.parse_notes_with_openai(
                content=request.content,
                extract_keywords=request.extract_keywords,
                extract_concepts=request.extract_concepts,
                extract_questions=request.extract_questions
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _structure_response(self, raw_result: Dict[str, Any], request: NotesParseRequest) -> Dict[str, Any]:
        """Structure the raw AI response into proper format"""
        parsed_content = raw_result.get("parsed_content", "")
        
        # Extract structured data from the parsed content
        structured_data = {
            "parsed_content": parsed_content,
            "keywords": [],
            "concepts": [],
            "study_questions": []
        }
        
        if request.extract_keywords:
            structured_data["keywords"] = self._extract_keywords_from_text(parsed_content)
        
        if request.extract_concepts:
            structured_data["concepts"] = self._extract_concepts_from_text(parsed_content)
        
        if request.extract_questions:
            structured_data["study_questions"] = self._extract_questions_from_text(parsed_content)
        
        return structured_data
    
    def _extract_keywords_from_text(self, text: str) -> List[KeywordExtraction]:
        """Extract keywords from AI response text"""
        keywords = []
        
        # Look for keyword patterns in the text
        keyword_patterns = [
            r"keywords?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"important terms?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"key words?:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                keyword_text = matches[0]
                # Extract individual keywords (assuming they're separated by commas or newlines)
                individual_keywords = re.split(r'[,\n]', keyword_text)
                
                for i, keyword in enumerate(individual_keywords[:10]):  # Limit to 10
                    clean_keyword = keyword.strip().strip('-').strip('*').strip()
                    if clean_keyword and len(clean_keyword) > 2:
                        # Assign decreasing importance scores
                        importance = max(0.5, 1.0 - (i * 0.1))
                        keywords.append(KeywordExtraction(
                            keyword=clean_keyword,
                            importance_score=importance
                        ))
                break
        
        return keywords[:10]  # Return max 10 keywords
    
    def _extract_concepts_from_text(self, text: str) -> List[ConceptExtraction]:
        """Extract concepts from AI response text"""
        concepts = []
        
        # Look for concept patterns
        concept_patterns = [
            r"concepts?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"key concepts?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"main ideas?:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                concept_text = matches[0]
                # Split into individual concepts
                individual_concepts = re.split(r'\n(?=\w)', concept_text)
                
                for i, concept in enumerate(individual_concepts[:8]):  # Limit to 8
                    clean_concept = concept.strip().strip('-').strip('*').strip()
                    if clean_concept and len(clean_concept) > 5:
                        # Try to separate concept name from definition
                        if ':' in clean_concept:
                            parts = clean_concept.split(':', 1)
                            concept_name = parts[0].strip()
                            definition = parts[1].strip()
                        else:
                            concept_name = clean_concept
                            definition = None
                        
                        importance = max(0.6, 1.0 - (i * 0.05))
                        concepts.append(ConceptExtraction(
                            concept=concept_name,
                            definition=definition,
                            importance_score=importance
                        ))
                break
        
        return concepts[:8]  # Return max 8 concepts
    
    def _extract_questions_from_text(self, text: str) -> List[StudyQuestion]:
        """Extract study questions from AI response text"""
        questions = []
        
        # Look for question patterns
        question_patterns = [
            r"questions?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"study questions?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"review questions?:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                question_text = matches[0]
                # Split into individual questions
                individual_questions = re.split(r'\n(?=\d+\.|\w)', question_text)
                
                for i, question in enumerate(individual_questions[:5]):  # Limit to 5
                    clean_question = question.strip().strip('-').strip('*').strip()
                    clean_question = re.sub(r'^\d+\.?\s*', '', clean_question)  # Remove numbering
                    
                    if clean_question and '?' in clean_question:
                        difficulties = ["easy", "medium", "hard"] #need to figure this properly
                        difficulty = difficulties[i % len(difficulties)]
                        
                        questions.append(StudyQuestion(
                            question=clean_question,
                            difficulty_level=difficulty,
                            question_type="short_answer"
                        ))
                break
        
        return questions[:5]  # Return max 5 questions


# Global notes parser agent instance
notes_parser_agent = NotesParserAgent(primary_provider="ollama")
