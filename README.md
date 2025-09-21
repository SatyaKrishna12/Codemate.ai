# Deep Researcher Agent

A sophisticated Python backend application for document processing and research using FastAPI, LangChain, and vector databases.

## Features

- 📄 **Document Processing**: Upload and process PDF, DOCX, TXT, and Markdown files
- 🔍 **Advanced Search**: Semantic, keyword, and hybrid search capabilities
- 🧠 **Research Assistant**: AI-powered question answering with source citations
- 🚀 **FastAPI Backend**: High-performance async API with automatic documentation
- 📊 **Vector Database**: FAISS-based similarity search for efficient retrieval
- 🔧 **Production Ready**: Comprehensive logging, error handling, and health checks

## Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd "Researcher Agent"

# Activate virtual environment
.\researcher_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (Optional)

Create a `.env` file in the project root to customize settings:

```env
# Application Settings
APP_NAME="Deep Researcher Agent"
DEBUG=true
ENVIRONMENT=development

# Server Settings
HOST=0.0.0.0
PORT=8000

# OpenAI API (Optional - for enhanced LLM capabilities)
OPENAI_API_KEY=your_openai_api_key_here

# Model Settings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# File Upload Settings
MAX_FILE_SIZE=52428800  # 50MB in bytes
```

### 3. Run the Application

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### Document Management
- `POST /api/v1/documents/upload` - Upload a single document
- `POST /api/v1/documents/upload-multiple` - Upload multiple documents
- `DELETE /api/v1/documents/{document_id}` - Delete a document
- `GET /api/v1/documents/stats` - Get document statistics

### Search
- `POST /api/v1/search/` - Search for relevant documents
- `GET /api/v1/search/stats` - Get search statistics

### Research
- `POST /api/v1/research/` - Ask research questions
- `GET /api/v1/research/stats` - Get research statistics

### Health & Monitoring
- `GET /api/v1/health/` - Basic health check
- `GET /api/v1/health/detailed` - Detailed system status
- `GET /api/v1/health/readiness` - Service readiness check
- `GET /api/v1/health/liveness` - Service liveness check

## Usage Examples

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"
```

### Search Documents

```bash
curl -X POST "http://localhost:8000/api/v1/search/" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning algorithms",
       "search_type": "semantic",
       "max_results": 5,
       "similarity_threshold": 0.7
     }'
```

### Research Questions

```bash
curl -X POST "http://localhost:8000/api/v1/research/" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What are the key benefits of transformer architectures?",
       "context_limit": 5,
       "include_sources": true
     }'
```

## Project Structure

```
Researcher Agent/
├── app/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Logging setup
│   │   └── exceptions.py      # Exception handling
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── routes/
│   │   ├── documents.py       # Document management routes
│   │   ├── search.py          # Search routes
│   │   ├── research.py        # Research routes
│   │   └── health.py          # Health check routes
│   ├── services/
│   │   ├── document_processor.py    # Document processing
│   │   ├── embedding_service.py     # Text embeddings
│   │   ├── vector_store.py          # Vector database
│   │   ├── query_processor.py       # Search processing
│   │   └── response_synthesizer.py  # Response generation
│   └── utils/
├── data/
│   ├── uploads/               # Uploaded documents
│   └── vectors/               # Vector database files
├── logs/                      # Application logs
├── researcher_env/            # Virtual environment
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Architecture

### Core Components

1. **Document Processor**: Extracts text and metadata from various file formats
2. **Embedding Service**: Generates vector embeddings using Sentence Transformers
3. **Vector Store**: FAISS-based similarity search and storage
4. **Query Processor**: Handles different types of search queries
5. **Response Synthesizer**: Generates comprehensive answers using RAG

### Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing LLM applications
- **Sentence Transformers**: State-of-the-art text embeddings
- **FAISS**: Efficient similarity search and clustering
- **Pydantic**: Data validation and settings management
- **Structlog**: Structured logging for better observability

## Development

### Adding New Document Types

1. Update `supported_file_types` in `config.py`
2. Add processing logic in `document_processor.py`
3. Update the `DocumentType` enum in `schemas.py`

### Customizing Search

1. Modify search algorithms in `query_processor.py`
2. Adjust embedding models in `embedding_service.py`
3. Configure similarity thresholds in settings

### Extending Research Capabilities

1. Add LLM integration in `response_synthesizer.py`
2. Customize prompts for different research scenarios
3. Implement domain-specific processing logic

## Production Deployment

### Environment Variables

Set the following environment variables for production:

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
OPENAI_API_KEY=your_production_key
```

### Health Monitoring

The application provides comprehensive health checks:
- `/api/v1/health/liveness` - Basic liveness probe
- `/api/v1/health/readiness` - Readiness for traffic
- `/api/v1/health/detailed` - Complete system status

### Scaling Considerations

- Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
- Implement horizontal scaling with load balancers
- Consider using a distributed vector database for large deployments
- Add caching layers for frequently accessed content

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Ensure internet connectivity for downloading Sentence Transformers models
2. **Large File Upload**: Adjust `MAX_FILE_SIZE` in configuration
3. **Memory Issues**: Monitor memory usage during document processing
4. **Vector Store Corruption**: Delete vector database files to rebuild

### Logging

Logs are stored in the `logs/` directory:
- `app.log` - General application logs
- `error.log` - Error-specific logs

Set `LOG_LEVEL=DEBUG` for detailed debugging information.

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive error handling and logging
3. Include unit tests for new functionality
4. Update documentation for API changes

## License

This project is open source and available under the MIT License.
