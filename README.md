# 🎓 AI Academic Assistant

A powerful **FastAPI-based AI assistant** specifically designed for academic content processing, featuring advanced **OCR for handwritten notes**, intelligent text extraction, and comprehensive analysis capabilities.

## ✨ Key Features

### 📝 **Advanced OCR for Handwritten Notes**
- **Multi-strategy OCR** with quality scoring for optimal text extraction
- **Advanced image preprocessing** using OpenCV and PIL
- **AI-powered text correction** using local LLM
- **Support for scanned PDFs** and handwritten academic content
- **Real-time quality assessment** and debugging output

### 🧠 **Intelligent Content Analysis**
- **Smart keyword extraction** from academic materials
- **Concept identification** with definitions from headings and topics
- **Automatic question extraction** from course materials
- **Comprehensive summarization** with multiple formats

### 📁 **Multi-Format File Support**
- **PDF**: Text-based and scanned/handwritten documents
- **DOCX**: Microsoft Word documents  
- **PPTX**: PowerPoint presentations
- **TXT**: Plain text files
- **Enhanced file handling** up to 100MB for large academic documents

### 🤖 **Dual AI System**
- **Primary**: Local Ollama (Mistral 7B) for privacy and speed
- **Fallback**: OpenAI API for reliability
- **Automatic failover** between providers

## 🚀 Quick Start

### 1. **Prerequisites**
- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for handwritten content
- [Poppler](https://poppler.freedesktop.org/) for PDF processing

### 2. **Installation**
```bash
# Clone the repository
git clone https://github.com/Abhyuday-06/AI-Academic-Assistant.git
cd AI-Academic-Assistant

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Environment Configuration**
Create a `.env` file with your configuration:
```env
# OpenAI Configuration (fallback)
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration (primary)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Application Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
SECRET_KEY=your_secret_key_here
LOG_LEVEL=INFO
```

### 4. **Setup Ollama & Dependencies**
```bash
# Install and start Ollama
ollama pull mistral:7b
ollama serve

# For Windows - Install Tesseract and Poppler:
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler: Download and extract to C:\poppler-xx.x.x\
```

### 5. **Run the Application**
```bash
# Start the development server
python run_dev.py

# Or manually with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. **Access Your Assistant**
- 📖 **API Documentation**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health
- 🏠 **API Root**: http://localhost:8000/

## 📚 API Endpoints

### 📝 **Notes Processing**
```http
POST /notes/parse-file
POST /notes/parse
```
**Features:**
- Upload handwritten notes, PDFs, DOCX, PPTX files
- Advanced OCR with multiple extraction strategies
- Extract keywords, concepts, and questions
- Real-time quality monitoring and text correction

### 📄 **Content Summarization**  
```http
POST /summarize/file
POST /summarize/
POST /summarize/bullet-points
POST /summarize/abstract
```
**Features:**
- Multiple summary formats (comprehensive, bullet points, abstracts)
- Customizable length and focus areas
- Academic-optimized summarization

### 🔍 **System Health**
```http
GET /health
GET /notes/health
GET /summarize/health
```

## 💡 Usage Examples

### **Upload and Process Handwritten Notes**
```python
import httpx

# Upload a handwritten PDF
with open("handwritten_notes.pdf", "rb") as file:
    response = httpx.post(
        "http://localhost:8000/notes/parse-file",
        files={"file": file},
        params={
            "extract_keywords": True,
            "extract_concepts": True,
            "extract_questions": True
        }
    )

result = response.json()
print(f"Keywords: {result['keywords']}")
print(f"Concepts: {result['concepts']}")
print(f"Questions: {result['study_questions']}")
```

### **Process Text Content**
```python
import httpx

response = httpx.post("http://localhost:8000/notes/parse", json={
    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data...",
    "extract_keywords": True,
    "extract_concepts": True,
    "extract_questions": False
})

result = response.json()
print(f"Analysis: {result['parsed_content']}")
```

### **Summarize Academic Content**
```python
import httpx

response = httpx.post("http://localhost:8000/summarize/", json={
    "content": "Your long academic content here...",
    "summary_type": "comprehensive",
    "max_length": 300,
    "focus_areas": ["key concepts", "main findings"]
})

summary = response.json()
print(f"Summary: {summary['summary']}")
```

## 🏗️ Project Architecture

```
AI-Academic-Assistant/
├── 📁 app/
│   ├── 🤖 agents/           # AI processing agents
│   │   ├── notes_parser.py  # Handwritten notes & content analysis
│   │   └── summarizer.py    # Content summarization
│   ├── 📊 models/           # Pydantic data models
│   ├── 🛣️ routers/          # FastAPI route handlers
│   │   ├── notes_router.py  # OCR & notes processing
│   │   └── summarizer_router.py
│   └── 🔧 utils/            # Core utilities
│       ├── config.py        # App configuration
│       ├── ollama_client.py # Local AI client
│       └── openai_client.py # Fallback AI client
├── 📂 local_files/         # File upload storage
├── 🐍 main.py              # Application entry point
├── ⚙️ run_dev.py           # Development runner
├── 📋 requirements.txt     # Python dependencies
└── 📖 README.md           # You are here!
```

## 🔧 Advanced Configuration

### **OCR Settings**
```python
# In .env or app/utils/config.py
OCR_DPI=300                              # Scan quality
OCR_PARALLEL_WORKERS=4                   # Processing threads
ENABLE_ADVANCED_OCR_PREPROCESSING=True   # Image enhancement
ENABLE_OCR_CORRECTION=True               # AI text correction
```

### **File Limits**
```python
MAX_FILE_SIZE_MB=100      # Large academic documents
MAX_CONTENT_LENGTH=500000 # Long research papers
```

## 🧪 Development & Testing

### **Code Quality**
```bash
# Format code
black app/
isort app/

# Type checking
mypy app/

# Linting
flake8 app/
```

### **Debugging OCR**
The system provides detailed OCR debugging output in the terminal:
```
🚀 ENHANCED ACADEMIC OCR SYSTEM
📄 Successfully converted 5 pages to images
📄 Processing page 1/5...
  📊 Academic Mixed Content: 0.85 score, 234 chars
  ✅ Page 1: 234 chars, score: 0.85
📊 FINAL RESULTS:
  📈 Success rate: 5/5 pages (100.0%)
  📊 Average quality score: 0.82
  📝 Total extracted text: 1200 characters
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local AI inference
- **OpenAI** for fallback AI capabilities  
- **Tesseract OCR** for text recognition
- **FastAPI** for the excellent web framework
- **OpenCV** for advanced image processing

---

**Built with ❤️ for students, researchers, and educators**
