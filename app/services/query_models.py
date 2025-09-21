"""
Intelligent Query Processing System using LangChain

This module provides a modular, lightweight query processing system that:
1. Analyzes queries to understand intent and complexity
2. Decomposes complex queries into manageable sub-questions
3. Uses LangChain's reasoning capabilities for execution
4. Maintains conversation context and memory
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid


class QueryType(Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"           # Direct fact-based questions
    COMPARATIVE = "comparative"    # Compare two or more entities
    ANALYTICAL = "analytical"      # Requires analysis and reasoning
    SUMMARIZATION = "summarization"  # Summarize information
    EXPLANATORY = "explanatory"    # Explain relationships or concepts
    UNKNOWN = "unknown"           # Cannot determine type


class ComplexityLevel(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single-step, direct answer
    MODERATE = "moderate"  # 2-3 reasoning steps
    COMPLEX = "complex"    # Multiple steps, dependencies
    VERY_COMPLEX = "very_complex"  # Requires extensive decomposition


@dataclass
class Entity:
    """Represents an extracted entity from the query."""
    text: str
    entity_type: str  # PERSON, ORG, CONCEPT, etc.
    confidence: float = 0.0


@dataclass
class Query:
    """Main query data model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_text: str = ""
    processed_text: str = ""
    query_type: QueryType = QueryType.UNKNOWN
    complexity: ComplexityLevel = ComplexityLevel.SIMPLE
    entities: List[Entity] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubQuery:
    """Sub-question derived from query decomposition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    parent_query_id: str = ""
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite sub-queries
    order: int = 0
    is_completed: bool = False
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """Individual step in the reasoning process."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    order: int = 0
    action: str = ""  # The action taken (search, analyze, compare, etc.)
    input_data: Any = None
    result: Any = None
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for query execution including conversation history."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    current_query: Optional[Query] = None
    sub_queries: List[SubQuery] = field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class QueryProcessingResult(BaseModel):
    """Result of query processing."""
    query_id: str
    original_query: str
    final_answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list)
    sub_queries: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    """Single turn in a conversation."""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str
    assistant_response: str
    query_result: Optional[QueryProcessingResult] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
