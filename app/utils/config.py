from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings"""
    
    # OpenAI Configuration
    openai_api_key: str
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"
    
    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    
    # Logging
    log_level: str = "INFO"
    
    # API Limits
    max_content_length: int = 500000  # Increased for large documents
    max_file_size_mb: int = 100  # Increased for large PDFs
    
    # OCR & Processing
    ocr_dpi: int = 300  # DPI for scanning PDFs
    ocr_parallel_workers: int = 4  # Number of parallel processes for OCR
    enable_advanced_ocr_preprocessing: bool = True # Use a more advanced image processing pipeline
    enable_ocr_correction: bool = True  # Use LLM to correct OCR output
    
    # Properties for backward compatibility and easy access
    @property
    def MAX_CONTENT_LENGTH(self) -> int:
        return self.max_content_length
    
    @property
    def MAX_FILE_SIZE(self) -> int:
        return self.max_file_size_mb * 1024 * 1024  # Convert MB to bytes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
