#!/usr/bin/env python3
"""
Development startup script for Academic Assistant API
"""
import uvicorn
from app.utils.config import settings

if __name__ == "__main__":
    print("Powering up AI Academic Assistant...")
    print(f"Server will run on: http://{settings.api_host}:{settings.api_port}")
    print(f"API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"Debug mode: {settings.debug}")
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"Ollama Model: {settings.ollama_model}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
