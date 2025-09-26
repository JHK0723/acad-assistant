import time
import json
from typing import Dict, Any
import structlog

from app.models import QuestionPaperRequest, QuestionPaperResponse
from app.utils.ollama_client import ollama_client
from app.utils.openai_client import openai_client

logger = structlog.get_logger()


class QuestionPaperAgent:
    """Agent for generating question papers based on syllabus and rules"""

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.initialized = False

    async def _initialize_provider(self):
        """Performs a health check on the primary provider and switches if it fails."""
        if self.provider == "openai":
            logger.info("Performing health check for primary provider: OpenAI")
            is_healthy = await openai_client.health_check()
            if not is_healthy:
                logger.warning("OpenAI health check failed. Switching to fallback provider: Ollama.")
                self.provider = "ollama"
            else:
                logger.info("OpenAI is healthy. Proceeding with OpenAI as the provider.")
        self.initialized = True

    async def generate_question_paper(self, request: QuestionPaperRequest) -> QuestionPaperResponse:
        """
        Main method to generate a question paper using the specified provider.
        """
        start_time = time.time()

        if not self.initialized:
            await self._initialize_provider()

        try:
            generation_prompt = self._create_generation_prompt(request)

            logger.info(f"Generating question paper using provider: {self.provider}")
            if self.provider == "openai":
                raw_response = await openai_client.generate_completion(
                    prompt=generation_prompt,
                    system_message="You are an expert exam paper generator. Follow the JSON output format strictly.",
                    temperature=0.4,
                    max_tokens=4000
                )
            else: # Fallback to ollama
                raw_response = await ollama_client.generate_completion(
                    prompt=generation_prompt,
                    system_message="You are an expert exam paper generator. Follow the JSON output format strictly.",
                    temperature=0.4,
                    max_tokens=4000
                )
            
            parsed_paper = self._parse_llm_response(raw_response)
            
            # Use Pydantic to validate and structure the final response
            response_data = QuestionPaperResponse(
                success=True,
                message="Question paper generated successfully",
                paper_data=parsed_paper,
                processing_time=time.time() - start_time,
                agent_used=self.provider
            )
            return response_data

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response", error=str(e), raw_response=raw_response)
            return QuestionPaperResponse(success=False, message=f"Failed to parse LLM response: {e}", paper_data=None, processing_time=time.time() - start_time, agent_used=self.provider)
        except Exception as e:
            logger.error("Question paper generation failed", error=str(e))
            return QuestionPaperResponse(success=False, message=f"An unexpected error occurred: {str(e)}", paper_data=None, processing_time=time.time() - start_time, agent_used="none")

    def _create_generation_prompt(self, request: QuestionPaperRequest) -> str:
        """
        Creates a detailed prompt for the LLM, mapping test_type to difficulty and title.
        """
        # Map test_type to difficulty
        difficulty_map = {
            "CAT-1": "easy",
            "CAT-2": "tough",
            "FAT": "medium"
        }
        test_type_upper = request.test_type.upper()
        # Use the mapped difficulty, ignoring the one in the request
        final_difficulty = difficulty_map.get(test_type_upper, "medium")
        # Use the test_type as the title, ignoring the one in the request
        final_title = test_type_upper

        prompt = f"""
        {{
            "version": "1.0",
            "desc": "Generates exam question papers based on syllabus, test type, modules, and difficulty. Handles difficulty balancing and solved-example logic for subdivision questions.",
            "instructions": "You are an expert exam paper generator. Follow these rules step-by-step: 1) Use input.syllabus, input.test_type, input.modules, and input.difficulty. 2) For CAT-1 and CAT-2: Create EXACTLY 5 questions, each 10 marks, total = 50 marks. For FAT: Create EXACTLY 10 questions, each 10 marks, total = 100 marks. 3) For each 10-mark question: - If NO subparts: Generate a single scenario-based question requiring deep analysis/application. - If EXACTLY 2 subparts of 5 marks each: Generate two direct formula-based questions, each from DIFFERENT modules. - If MORE THAN 2 subparts: Generate multiple short direct questions whose combined total = 10 marks. 4) Subdivision questions can be generated by SLIGHTLY altering solved examples in the syllabus, but ONLY if a previous no-subdivision question is very hard/complex OR difficulty balance rules require it. 5) Difficulty balancing: - If 50% or more of no-subdivision questions are EASY, all subdivision questions must be TOUGH. - If 50% or more of no-subdivision questions are TOUGH, all subdivision questions must be EASY by using solved examples. - If balanced, follow global input.difficulty. 6) Spread questions evenly across modules. If modules > number of questions, prioritize high-weight topics in syllabus. 7) Difficulty mapping: easy = direct/computational, medium = conceptual + computation, hard = multi-step complex analytical questions. 8) Output strictly in the given JSON format, no extra text or commentary. 9) If syllabus.text is empty or syllabus.uploaded = false, use only modules and add a note in metadata.notes.",
            "input": {{
                "syllabus": {{
                    "text": {json.dumps(request.syllabus_text)},
                    "uploaded": {json.dumps(bool(request.syllabus_text))}
                }},
                "test_type": {json.dumps(test_type_upper)},
                "modules": {json.dumps(request.modules)},
                "difficulty": {json.dumps(final_difficulty)},
                "title": {json.dumps(final_title)}
            }},
            "output_schema": {{
                "metadata": {{"title": "string", "test_type": "string", "modules": ["string"], "difficulty": "string", "total_marks": "integer", "notes": "string (optional)"}},
                "paper": [
                  {{"q_no": "integer", "marks": "integer", "parts": [{{"label": "string", "marks": "integer", "text": "string", "module": ["string"], "difficulty": "string"}}], "instructions": "string (optional)"}}
                ],
                "validation": {{"total_marks_check": "integer", "unique_questions": "boolean"}}
            }}
        }}
        """
        return prompt.strip()

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the JSON string from the LLM into a dictionary.
        """
        try:
            # Clean potential markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error("JSON decoding failed", error=str(e), response_text=response_text)
            raise

question_paper_agent = QuestionPaperAgent(provider="openai")


