"""
Context Management Component

Manages conversation history, user preferences, and session state
for multi-turn interactions using LangChain memory capabilities.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

try:
    from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
    from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .query_models import ExecutionContext, ConversationTurn, Query, QueryProcessingResult

# Use simple logging if app.core.logging not available
try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context, memory, and session state for intelligent query processing.
    Provides seamless multi-turn conversation capabilities.
    """
    
    def __init__(self, max_token_limit: int = 4000, summary_threshold: int = 2000):
        """
        Initialize the context manager.
        
        Args:
            max_token_limit: Maximum tokens to keep in memory
            summary_threshold: Token threshold for creating summaries
        """
        self.max_token_limit = max_token_limit
        self.summary_threshold = summary_threshold
        self.sessions: Dict[str, ExecutionContext] = {}
        self.memory_instances: Dict[str, Any] = {}
        
        logger.info("ContextManager initialized")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional session ID, generates one if not provided
            
        Returns:
            Session ID
        """
        if not session_id:
            session_id = f"session_{datetime.utcnow().timestamp()}"
        
        # Create execution context
        context = ExecutionContext(session_id=session_id)
        self.sessions[session_id] = context
        
        # Initialize LangChain memory if available
        if LANGCHAIN_AVAILABLE:
            self.memory_instances[session_id] = ConversationSummaryBufferMemory(
                max_token_limit=self.max_token_limit,
                return_messages=True,
                memory_key="chat_history",
                input_key="input",
                output_key="output"
            )
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ExecutionContext]:
        """Get session context by ID."""
        return self.sessions.get(session_id)
    
    def update_session_context(self, session_id: str, query: Query, 
                             result: QueryProcessingResult) -> None:
        """
        Update session context with new query and result.
        
        Args:
            session_id: Session identifier
            query: The processed query
            result: Query processing result
        """
        context = self.sessions.get(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found")
            return
        
        # Update current query
        context.current_query = query
        context.last_updated = datetime.utcnow()
        
        # Add to conversation history
        turn = ConversationTurn(
            user_input=query.original_text,
            assistant_response=result.final_answer,
            query_result=result
        )
        
        context.conversation_history.append({
            'turn_id': turn.turn_id,
            'user_input': turn.user_input,
            'assistant_response': turn.assistant_response,
            'timestamp': turn.timestamp.isoformat(),
            'metadata': result.metadata
        })
        
        # Update LangChain memory if available
        if session_id in self.memory_instances:
            memory = self.memory_instances[session_id]
            memory.save_context(
                {"input": query.original_text},
                {"output": result.final_answer}
            )
        
        logger.info(f"Updated context for session {session_id}")
    
    def get_conversation_history(self, session_id: str, 
                               last_n_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            last_n_turns: Number of recent turns to return (all if None)
            
        Returns:
            List of conversation turns
        """
        context = self.sessions.get(session_id)
        if not context:
            return []
        
        history = context.conversation_history
        if last_n_turns:
            history = history[-last_n_turns:]
        
        return history
    
    def get_memory_variables(self, session_id: str) -> Dict[str, Any]:
        """Get LangChain memory variables for a session."""
        if session_id not in self.memory_instances:
            return {}
        
        memory = self.memory_instances[session_id]
        return memory.load_memory_variables({})
    
    def analyze_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze conversation context to extract insights and patterns.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context analysis results
        """
        context = self.sessions.get(session_id)
        if not context:
            return {}
        
        history = context.conversation_history
        if not history:
            return {'turn_count': 0}
        
        # Basic statistics
        turn_count = len(history)
        total_user_words = sum(len(turn['user_input'].split()) for turn in history)
        total_assistant_words = sum(len(turn['assistant_response'].split()) for turn in history)
        
        # Extract query types and topics
        query_types = []
        topics = []
        
        for turn in history:
            metadata = turn.get('metadata', {})
            if 'query_type' in metadata:
                query_types.append(metadata['query_type'])
        
        # Find common themes
        user_inputs = [turn['user_input'].lower() for turn in history]
        common_words = self._extract_common_words(user_inputs)
        
        # Detect follow-up patterns
        follow_up_indicators = self._detect_follow_ups(user_inputs)
        
        return {
            'turn_count': turn_count,
            'total_user_words': total_user_words,
            'total_assistant_words': total_assistant_words,
            'avg_user_words_per_turn': total_user_words / turn_count if turn_count > 0 else 0,
            'avg_assistant_words_per_turn': total_assistant_words / turn_count if turn_count > 0 else 0,
            'query_types': query_types,
            'common_topics': common_words,
            'follow_up_percentage': follow_up_indicators['percentage'],
            'conversation_duration_minutes': self._calculate_duration(history),
            'session_created': context.created_at.isoformat(),
            'last_updated': context.last_updated.isoformat()
        }
    
    def detect_follow_up_intent(self, session_id: str, new_query: str) -> Dict[str, Any]:
        """
        Detect if a new query is a follow-up to previous conversation.
        
        Args:
            session_id: Session identifier
            new_query: New query text
            
        Returns:
            Follow-up analysis results
        """
        context = self.sessions.get(session_id)
        if not context or not context.conversation_history:
            return {'is_follow_up': False, 'confidence': 0.0}
        
        last_turn = context.conversation_history[-1]
        last_user_input = last_turn['user_input'].lower()
        new_query_lower = new_query.lower()
        
        # Follow-up indicators
        follow_up_words = [
            'also', 'and', 'what about', 'how about', 'can you also',
            'additionally', 'furthermore', 'more', 'other', 'another',
            'that', 'this', 'it', 'they', 'them'
        ]
        
        # Pronouns and demonstratives (indicating reference to previous context)
        reference_words = ['that', 'this', 'it', 'they', 'them', 'these', 'those']
        
        # Calculate similarity and follow-up indicators
        has_follow_up_words = any(word in new_query_lower for word in follow_up_words)
        has_reference_words = any(word in new_query_lower for word in reference_words)
        
        # Short queries are often follow-ups
        is_short = len(new_query.split()) <= 5
        
        # Calculate confidence
        confidence = 0.0
        if has_follow_up_words:
            confidence += 0.4
        if has_reference_words:
            confidence += 0.3
        if is_short:
            confidence += 0.2
        
        # Check for topic continuity
        topic_similarity = self._calculate_topic_similarity(last_user_input, new_query_lower)
        confidence += topic_similarity * 0.3
        
        is_follow_up = confidence > 0.5
        
        return {
            'is_follow_up': is_follow_up,
            'confidence': min(confidence, 1.0),
            'indicators': {
                'has_follow_up_words': has_follow_up_words,
                'has_reference_words': has_reference_words,
                'is_short': is_short,
                'topic_similarity': topic_similarity
            },
            'previous_context': last_turn['user_input'][:100] + '...' if len(last_turn['user_input']) > 100 else last_turn['user_input']
        }
    
    def get_relevant_context(self, session_id: str, query: str, 
                           max_context_turns: int = 3) -> str:
        """
        Get relevant context from conversation history for the current query.
        
        Args:
            session_id: Session identifier
            query: Current query
            max_context_turns: Maximum number of previous turns to consider
            
        Returns:
            Relevant context string
        """
        history = self.get_conversation_history(session_id, max_context_turns)
        if not history:
            return ""
        
        # Build context string
        context_parts = []
        for turn in history:
            context_parts.append(f"User: {turn['user_input']}")
            context_parts.append(f"Assistant: {turn['assistant_response'][:200]}...")
        
        return "\n".join(context_parts)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old sessions to manage memory usage.
        
        Args:
            max_age_hours: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        old_sessions = []
        
        for session_id, context in self.sessions.items():
            if context.last_updated < cutoff_time:
                old_sessions.append(session_id)
        
        # Remove old sessions
        for session_id in old_sessions:
            del self.sessions[session_id]
            if session_id in self.memory_instances:
                del self.memory_instances[session_id]
        
        logger.info(f"Cleaned up {len(old_sessions)} old sessions")
        return len(old_sessions)
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data for backup or analysis."""
        context = self.sessions.get(session_id)
        if not context:
            return None
        
        return {
            'session_id': session_id,
            'created_at': context.created_at.isoformat(),
            'last_updated': context.last_updated.isoformat(),
            'conversation_history': context.conversation_history,
            'user_preferences': context.user_preferences,
            'analysis': self.analyze_conversation_context(session_id)
        }
    
    def _extract_common_words(self, texts: List[str], min_length: int = 3) -> List[str]:
        """Extract common words from a list of texts."""
        word_counts = {}
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for text in texts:
            words = text.split()
            for word in words:
                clean_word = word.strip('.,!?').lower()
                if len(clean_word) >= min_length and clean_word not in stop_words:
                    word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        # Return words that appear in multiple texts
        threshold = max(1, len(texts) // 2)
        return [word for word, count in word_counts.items() if count >= threshold]
    
    def _detect_follow_ups(self, user_inputs: List[str]) -> Dict[str, Any]:
        """Detect follow-up patterns in user inputs."""
        if len(user_inputs) < 2:
            return {'count': 0, 'percentage': 0.0}
        
        follow_up_count = 0
        follow_up_indicators = ['also', 'and', 'what about', 'how about', 'more', 'other']
        
        for i in range(1, len(user_inputs)):
            current = user_inputs[i]
            if any(indicator in current for indicator in follow_up_indicators):
                follow_up_count += 1
        
        percentage = (follow_up_count / (len(user_inputs) - 1)) * 100
        return {'count': follow_up_count, 'percentage': percentage}
    
    def _calculate_duration(self, history: List[Dict[str, Any]]) -> float:
        """Calculate conversation duration in minutes."""
        if len(history) < 2:
            return 0.0
        
        try:
            start_time = datetime.fromisoformat(history[0]['timestamp'])
            end_time = datetime.fromisoformat(history[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0.0
    
    def _calculate_topic_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple topic similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
