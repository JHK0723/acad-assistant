import time
import re
import json
from typing import Dict, Any, List, Optional
import structlog
from app.utils.ollama_client import ollama_client
from app.models import (
    NotesParseRequest, NotesParseResponse, SummarizeRequest, SummaryResponse,
    KeywordExtraction, ConceptExtraction, StudyQuestion
)

logger = structlog.get_logger()


class AcademicAgent:
    """Unified agent for academic content processing - parsing and summarizing"""
    
    def __init__(self, primary_provider: str = "ollama"):
        self.primary_provider = primary_provider
        
    async def parse_and_summarize(self, request: NotesParseRequest) -> NotesParseResponse:
        """
        Main method to parse notes and generate summary using Ollama Mistral 7B
        This combines parsing functionality with summarization in a single call
        """
        start_time = time.time()
        
        try:
            # Step 1: Parse the content to extract structured information
            parsed_result = await self._parse_with_ollama(request)
            
            # Step 2: Generate summary using the parsed content and original content
            summary = await self._summarize_parsed_content(
                original_content=request.content,
                parsed_content=parsed_result.get("parsed_content", ""),
                keywords=[kw.keyword for kw in parsed_result.get("keywords", [])],
                concepts=[c.concept for c in parsed_result.get("concepts", [])]
            )
            
            # Step 3: Combine results
            processing_time = time.time() - start_time
            
            return NotesParseResponse(
                success=True,
                message="Content parsed and summarized successfully",
                parsed_content=summary,  # Return the summary as the main content
                keywords=parsed_result.get("keywords", []),
                concepts=parsed_result.get("concepts", []),
                study_questions=parsed_result.get("study_questions", []),
                processing_time=processing_time,
                agent_used="ollama_unified"
            )
            
        except Exception as e:
            logger.error("Unified parsing and summarization failed", error=str(e))
            return NotesParseResponse(
                success=False,
                message=f"Processing failed: {str(e)}",
                parsed_content="",
                processing_time=time.time() - start_time,
                agent_used="none"
            )
    
    async def parse_only(self, request: NotesParseRequest) -> NotesParseResponse:
        """Parse notes without summarization - for backward compatibility"""
        start_time = time.time()
        
        try:
            result = await self._parse_with_ollama(request)
            
            return NotesParseResponse(
                success=True,
                message="Notes parsed successfully",
                parsed_content=result.get("parsed_content", ""),
                keywords=result.get("keywords", []),
                concepts=result.get("concepts", []),
                study_questions=result.get("study_questions", []),
                processing_time=time.time() - start_time,
                agent_used="ollama"
            )
            
        except Exception as e:
            logger.error("Parsing failed", error=str(e))
            return NotesParseResponse(
                success=False,
                message=f"Parsing failed: {str(e)}",
                parsed_content="",
                processing_time=time.time() - start_time,
                agent_used="none"
            )
    
    async def summarize_only(self, request: SummarizeRequest) -> SummaryResponse:
        """Summarize content only - for backward compatibility"""
        start_time = time.time()
        original_length = len(request.content.split())
        
        try:
            # Use Ollama to generate summary
            result = await ollama_client.summarize_with_ollama(
                content=request.content,
                summary_type=request.summary_type,
                max_length=request.max_length,
                focus_areas=request.focus_areas
            )
            
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
                processing_time=time.time() - start_time,
                agent_used="ollama"
            )
            
        except Exception as e:
            logger.error("Summarization failed", error=str(e))
            return SummaryResponse(
                success=False,
                message=f"Summarization failed: {str(e)}",
                summary="",
                original_length=original_length,
                summary_length=0,
                compression_ratio=0.0,
                processing_time=time.time() - start_time,
                agent_used="none"
            )
    
    async def _parse_with_ollama(self, request: NotesParseRequest) -> Dict[str, Any]:
        """Parse content using Ollama and structure the response"""
        # Call Ollama for parsing
        raw_result = await ollama_client.parse_notes_with_ollama(
            content=request.content,
            extract_keywords=request.extract_keywords,
            extract_concepts=request.extract_concepts,
            extract_questions=request.extract_questions
        )
        
        parsed_content = raw_result.get("parsed_content", "")
        
        # Structure the response
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
    
    async def _summarize_parsed_content(self, original_content: str, parsed_content: str, 
                                      keywords: List[str], concepts: List[str]) -> str:
        """
        Generate a summary of the parsed content using Ollama Mistral 7B
        This is the key integration point between parsing and summarizing
        """
        
        # Create a comprehensive prompt that uses both original and parsed content
        system_message = """You are an expert academic assistant. You have been given original content and its parsed analysis. Create a comprehensive, well-structured summary that incorporates the key insights from the analysis."""
        
        # Build the summarization prompt
        prompt_parts = [
            "Create a comprehensive academic summary using the following information:",
            f"\nOriginal Content:\n{original_content}",  # No context limit
            f"\nParsed Analysis:\n{parsed_content}",
        ]
        
        if keywords:
            prompt_parts.append(f"\nKey Terms: {', '.join(keywords[:10])}")
        
        if concepts:
            prompt_parts.append(f"\nMain Concepts: {', '.join(concepts[:8])}")
        
        prompt_parts.extend([
            "\nRequirements:",
            "- Create a clear, comprehensive summary in bullet point format",
            "- Use bullet points (•) for main points and sub-bullets for details",
            "- Incorporate the key terms and concepts identified",
            "- Structure the content logically with clear sections",
            "- Target length: 400-600 words",
            "- Use academic writing style",
            "\nProvide a well-structured bullet point summary:"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = await ollama_client.generate_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,
                max_tokens=1200  # Increased for better comprehensive summaries
            )
            return response
        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            # Fallback to parsed content if summarization fails
            return parsed_content or "Summary generation failed."
    
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
                # Extract individual keywords
                individual_keywords = re.split(r'[,\n]', keyword_text)
                
                for i, keyword in enumerate(individual_keywords[:10]):
                    clean_keyword = keyword.strip().strip('-').strip('*').strip()
                    if clean_keyword and len(clean_keyword) > 2:
                        importance = max(0.5, 1.0 - (i * 0.1))
                        keywords.append(KeywordExtraction(
                            keyword=clean_keyword,
                            importance_score=importance
                        ))
                break
        
        return keywords[:10]
    
    def _extract_concepts_from_text(self, text: str) -> List[ConceptExtraction]:
        """Extract concepts from AI response text"""
        concepts = []
        
        concept_patterns = [
            r"concepts?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"key concepts?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"main ideas?:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                concept_text = matches[0]
                individual_concepts = re.split(r'\n(?=\w)', concept_text)
                
                for i, concept in enumerate(individual_concepts[:8]):
                    clean_concept = concept.strip().strip('-').strip('*').strip()
                    if clean_concept and len(clean_concept) > 5:
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
        
        return concepts[:8]
    
    def _extract_questions_from_text(self, text: str) -> List[StudyQuestion]:
        """Extract study questions from AI response text"""
        questions = []
        
        question_patterns = [
            r"questions?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"study questions?:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"review questions?:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                question_text = matches[0]
                individual_questions = re.split(r'\n(?=\d+\.|\w)', question_text)
                
                for i, question in enumerate(individual_questions[:5]):
                    clean_question = question.strip().strip('-').strip('*').strip()
                    clean_question = re.sub(r'^\d+\.?\s*', '', clean_question)
                    
                    if clean_question and '?' in clean_question:
                        difficulties = ["easy", "medium", "hard"]
                        difficulty = difficulties[i % len(difficulties)]
                        
                        questions.append(StudyQuestion(
                            question=clean_question,
                            difficulty_level=difficulty,
                            question_type="short_answer"
                        ))
                break
        
        return questions[:5]
    
    def _extract_key_points(self, summary: str, summary_type: str) -> List[str]:
        """Extract key points from summary text"""
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
            # For comprehensive summaries, extract sentences
            sentences = re.split(r'[.!?]+', summary)
            sorted_sentences = sorted(sentences, key=len, reverse=True)
            key_points = [s.strip() for s in sorted_sentences[:6] if s.strip() and len(s.strip()) > 20]
        
        return key_points[:8]


# Global academic agent instance
academic_agent = AcademicAgent(primary_provider="ollama")
