"""
Core configuration settings for the Researcher Agent application.
Uses Pydantic settings for environment variable management.
"""

import os
from typing import Optional, Dict
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="Deep Researcher Agent", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development/production)")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=True, description="Auto-reload on code changes")
    
    # API settings
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Database and storage paths
    data_dir: str = Field(default="./data", description="Data directory path")
    uploads_dir: str = Field(default="./data/uploads", description="Uploads directory")
    vectors_dir: str = Field(default="./data/vectors", description="Vector database directory")
    logs_dir: str = Field(default="./logs", description="Logs directory")
    
    # AI/ML Model settings
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    embedding_model_cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory"
    )
    max_chunk_size: int = Field(default=1000, description="Maximum text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    
    # Vector database settings
    vector_index_name: str = Field(default="researcher_index", description="FAISS index name")
    similarity_threshold: float = Field(default=0.7, description="Similarity search threshold")
    max_search_results: int = Field(default=10, description="Maximum search results")
    
    # IR System - Retrieval Configuration
    multi_query_retriever_queries: int = Field(default=3, description="Number of queries for MultiQueryRetriever")
    ensemble_retriever_weights: list[float] = Field(default=[0.7, 0.3], description="Weights for EnsembleRetriever [semantic, keyword]")
    contextual_compression_enabled: bool = Field(default=True, description="Enable contextual compression")
    
    # IR System - Ranking Configuration  
    ranking_weights: Dict[str, float] = Field(
        default={
            "semantic": 0.6,
            "recency": 0.2, 
            "credibility": 0.15,
            "user_preference": 0.05
        },
        description="Weights for multi-factor ranking"
    )
    relevance_threshold: float = Field(default=0.5, description="Minimum relevance score for results")
    
    # IR System - Aggregation & Deduplication
    deduplication_threshold: float = Field(default=0.95, description="Similarity threshold for deduplication")
    max_duplicate_sources: int = Field(default=3, description="Maximum sources per duplicate cluster")
    
    # IR System - Validation & Cross-referencing
    cross_reference_enabled: bool = Field(default=True, description="Enable cross-reference validation")
    consensus_threshold: float = Field(default=0.7, description="Threshold for consensus scoring")
    conflict_detection_enabled: bool = Field(default=True, description="Enable conflict detection")
    
    # IR System - Post-processing
    snippet_max_chars: int = Field(default=300, description="Maximum characters in snippets")
    max_quotes_per_result: int = Field(default=3, description="Maximum quotes per result")
    highlight_query_terms: bool = Field(default=True, description="Enable query term highlighting")
    
    # LangChain settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_project: Optional[str] = Field(default=None, description="LangSmith project name")
    
    # LLM and Synthesis Settings
    groq_api_key: str = Field(default="your_groq_api_key_placeholder", description="Groq API key")
    groq_model_name: str = Field(default="mixtral-8x7b-32768", description="Groq model name")
    max_context_tokens: int = Field(default=32000, description="Maximum context tokens for LLM")
    default_k: int = Field(default=10, description="Default number of documents to retrieve")
    response_timeout: int = Field(default=30, description="LLM response timeout in seconds")
    citation_style_default: str = Field(default="APA", description="Default citation style")
    llm_temperature: float = Field(default=0.1, description="LLM temperature for generation")
    llm_max_tokens: int = Field(default=4000, description="Maximum tokens for LLM output")
    
    # Content synthesis settings
    enable_contextual_compression: bool = Field(default=True, description="Enable contextual compression")
    synthesis_consensus_threshold: float = Field(default=0.7, description="Consensus threshold for assertions")
    assertion_min_sources: int = Field(default=2, description="Minimum sources required per assertion")
    max_sections: int = Field(default=10, description="Maximum sections in detailed reports")
    
    # Document processing settings
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max file size in bytes (50MB)")
    supported_file_types: list[str] = Field(
        default=[".pdf", ".txt", ".docx", ".md"],
        description="Supported file extensions"
    )
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """Initialize settings and create directories."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        directories = [
            self.data_dir,
            self.uploads_dir,
            self.vectors_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global settings instance
settings = Settings()
