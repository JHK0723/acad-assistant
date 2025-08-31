# Academic Assistant API

A FastAPI-based application that provides AI-powered academic assistance with two main agents:

## Features

- **NotesParserAgent**: Parses educational notes and extracts structured information
  - Keyword extraction with importance scoring
  - Concept identification with definitions
  - Study question generation
  - Support for multiple file formats (PDF, DOCX, TXT, MD)

- **SummarizerAgent**: Summarizes educational content in various formats
  - Bullet point summaries
  - Comprehensive summaries
  - Academic abstracts
  - Customizable summary length and focus areas

## AI Providers

- **Primary**: Ollama (Mistral 7B) - Local inference
- **Fallback**: OpenAI SDK - Cloud-based inference
- Automatic failover between providers for reliability

## Technology Stack

- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation and serialization
- **OpenAI SDK**: For GPT model integration
- **Ollama**: For local Mistral 7B inference
- **Structured Logging**: Using structlog
- **Docker**: Containerization support

## API Endpoints

### Notes Parser
- `POST /notes/parse` - Parse text content
- `POST /notes/parse-file` - Parse uploaded file
- `GET /notes/health` - Health check

### Summarizer
- `POST /summarize/` - Summarize content
- `POST /summarize/file` - Summarize uploaded file
- `POST /summarize/bullet-points` - Generate bullet summary
- `POST /summarize/abstract` - Generate academic abstract
- `GET /summarize/health` - Health check

### General
- `GET /` - API information
- `GET /health` - Overall health check
- `GET /docs` - Interactive API documentation

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Update the following variables in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: mistral:7b)
- `SECRET_KEY`: Secret key for security

### 3. Ollama Setup
Make sure Ollama is installed and running with Mistral 7B:
```bash
# Install Ollama (if not already installed)
# Download from: https://ollama.ai

# Pull Mistral 7B model
ollama pull mistral:7b

# Start Ollama server
ollama serve
```

### 4. Run the Application
```bash
# Development mode
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the API
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- API Root: http://localhost:8000/

## Usage Examples

### Parse Notes
```python
import httpx

response = httpx.post("http://localhost:8000/notes/parse", json={
    "content": "Machine learning is a subset of artificial intelligence...",
    "extract_keywords": True,
    "extract_concepts": True,
    "extract_questions": True
})
```

### Summarize Content
```python
import httpx

response = httpx.post("http://localhost:8000/summarize/", json={
    "content": "Long educational content here...",
    "summary_type": "comprehensive",
    "max_length": 500
})
```

## Development

### Project Structure
```
acad-assistant/
├── app/
│   ├── agents/          # AI agents (NotesParser, Summarizer)
│   ├── models/          # Pydantic models for data validation
│   ├── routers/         # FastAPI routers
│   └── utils/           # Utilities (config, clients)
├── tests/               # Test files
├── requirements.txt     # Dependencies
├── main.py             # Application entry point
└── README.md           # This file
```

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app
```

### Code Quality
```bash
# Format code
black app/

# Sort imports
isort app/

# Lint code
flake8 app/
```

## Docker Support

See Docker section below for containerization instructions.

## License

This project is licensed under the MIT License.
