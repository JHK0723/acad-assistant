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
    max_content_length: int = 100000  # Maximum content length in characters
    max_file_size_mb: int = 10  # Maximum file size in MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
