"""
Reasoning and Execution Component

Uses LangChain's capabilities for multi-step reasoning and execution.
Manages the execution of query plans with proper dependency handling.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict

# Optional LangChain imports
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.agents.format_scratchpad import format_to_openai_function_messages
    from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
    from langchain.tools import Tool
    from langchain.schema import AgentAction, AgentFinish
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .query_models import Query, SubQuery, ReasoningStep, ExecutionContext, QueryProcessingResult

# Use simple logging if app.core.logging not available
try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    Executes queries using LangChain's reasoning capabilities.
    Manages multi-step execution with proper dependency handling.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the query executor."""
        self.openai_api_key = openai_api_key
        self.llm = None
        self.agent_executor = None
        self.memory = None
        
        if LANGCHAIN_AVAILABLE and openai_api_key:
            self._init_langchain_components()
        else:
            logger.warning("LangChain not available or no API key provided, using mock executor")
        
        logger.info("QueryExecutor initialized")
    
    def _init_langchain_components(self):
        """Initialize LangChain components."""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            )
            
            # Create tools
            tools = self._create_tools()
            
            # Create agent
            prompt = self._create_agent_prompt()
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate"
            )
            
            logger.info("LangChain components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {e}")
            self.llm = None
            self.agent_executor = None
    
    def _create_tools(self) -> List[Any]:
        """Create tools for the agent."""
        if not LANGCHAIN_AVAILABLE:
            return []
            
        if not LANGCHAIN_AVAILABLE:
            return []
            
        return [
            Tool(
                name="search_information",
                description="Search for information about a specific topic or question",
                func=self._search_information
            ),
            Tool(
                name="analyze_data",
                description="Analyze and process information to draw conclusions",
                func=self._analyze_data
            ),
            Tool(
                name="compare_entities",
                description="Compare two or more entities and highlight differences",
                func=self._compare_entities
            ),
            Tool(
                name="summarize_information",
                description="Summarize complex information into key points",
                func=self._summarize_information
            )
        ]
    
    def _create_agent_prompt(self) -> Any:
        """Create the agent prompt template."""
        if not LANGCHAIN_AVAILABLE:
            return None
            
        return ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent query processing assistant. Your role is to:

1. Break down complex questions into manageable steps
2. Use the available tools to gather and process information
3. Reason through multi-step problems systematically
4. Provide clear, well-structured answers

Available tools:
- search_information: Find information about topics
- analyze_data: Analyze information and draw conclusions
- compare_entities: Compare different entities or concepts
- summarize_information: Create summaries of complex topics

Always explain your reasoning process and cite your sources when possible."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    async def execute_query(self, query: Query, sub_queries: List[SubQuery], 
                          context: ExecutionContext) -> QueryProcessingResult:
        """
        Execute a query with its sub-queries using LangChain.
        
        Args:
            query: The main query to execute
            sub_queries: List of sub-queries to execute
            context: Execution context with conversation history
            
        Returns:
            QueryProcessingResult with the final answer and reasoning steps
        """
        start_time = time.time()
        reasoning_steps = []
        
        try:
            # If no LangChain, use mock execution
            if not self.agent_executor:
                return await self._mock_execute_query(query, sub_queries, context)
            
            # Execute sub-queries in dependency order
            sub_query_results = {}
            if sub_queries:
                sub_query_results = await self._execute_sub_queries(sub_queries, reasoning_steps)
            
            # Execute main query with sub-query results
            final_answer = await self._execute_main_query(query, sub_query_results, reasoning_steps)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            
            # Create result
            result = QueryProcessingResult(
                query_id=query.id,
                original_query=query.original_text,
                final_answer=final_answer,
                confidence=0.8,  # Could be calculated based on tool confidence
                reasoning_steps=[asdict(step) for step in reasoning_steps],
                sub_queries=[asdict(sq) for sq in sub_queries],
                execution_time_ms=execution_time,
                sources=[],  # Could be extracted from tool outputs
                metadata={
                    'query_type': query.query_type.value,
                    'complexity': query.complexity.value,
                    'sub_query_count': len(sub_queries)
                }
            )
            
            logger.info(f"Executed query {query.id} in {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query {query.id}: {e}")
            return QueryProcessingResult(
                query_id=query.id,
                original_query=query.original_text,
                final_answer=f"Error executing query: {str(e)}",
                confidence=0.0,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _execute_sub_queries(self, sub_queries: List[SubQuery], 
                                 reasoning_steps: List[ReasoningStep]) -> Dict[str, str]:
        """Execute sub-queries in dependency order."""
        results = {}
        
        # Create execution order based on dependencies
        execution_order = self._create_execution_order(sub_queries)
        
        for sub_query in execution_order:
            step_start = time.time()
            
            # Prepare input with dependency results
            input_text = sub_query.text
            if sub_query.dependencies:
                dependency_context = "\n".join([
                    f"Previous result: {results.get(dep_id, 'No result')}"
                    for dep_id in sub_query.dependencies
                ])
                input_text = f"{input_text}\n\nContext from previous steps:\n{dependency_context}"
            
            # Execute using agent
            try:
                response = await asyncio.to_thread(
                    self.agent_executor.run,
                    input=input_text
                )
                results[sub_query.id] = response
                sub_query.result = response
                sub_query.is_completed = True
                
                # Record reasoning step
                step = ReasoningStep(
                    description=f"Executed sub-query: {sub_query.text}",
                    order=len(reasoning_steps) + 1,
                    action="sub_query_execution",
                    input_data=input_text,
                    result=response,
                    confidence=0.8,
                    execution_time_ms=(time.time() - step_start) * 1000,
                    metadata={'sub_query_id': sub_query.id}
                )
                reasoning_steps.append(step)
                
            except Exception as e:
                logger.error(f"Error executing sub-query {sub_query.id}: {e}")
                results[sub_query.id] = f"Error: {str(e)}"
        
        return results
    
    async def _execute_main_query(self, query: Query, sub_query_results: Dict[str, str],
                                reasoning_steps: List[ReasoningStep]) -> str:
        """Execute the main query with sub-query results."""
        step_start = time.time()
        
        # Prepare input with sub-query results
        input_text = query.original_text
        if sub_query_results:
            context = "\n".join([
                f"Sub-query result {i+1}: {result}"
                for i, result in enumerate(sub_query_results.values())
            ])
            input_text = f"{input_text}\n\nAvailable information:\n{context}\n\nPlease provide a comprehensive answer based on this information."
        
        # Execute main query
        try:
            response = await asyncio.to_thread(
                self.agent_executor.run,
                input=input_text
            )
            
            # Record reasoning step
            step = ReasoningStep(
                description="Executed main query with sub-query context",
                order=len(reasoning_steps) + 1,
                action="main_query_execution",
                input_data=input_text,
                result=response,
                confidence=0.9,
                execution_time_ms=(time.time() - step_start) * 1000,
                metadata={'query_id': query.id}
            )
            reasoning_steps.append(step)
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing main query {query.id}: {e}")
            return f"Error executing main query: {str(e)}"
    
    def _create_execution_order(self, sub_queries: List[SubQuery]) -> List[SubQuery]:
        """Create execution order based on dependencies (topological sort)."""
        # Simple topological sort
        result = []
        remaining = sub_queries.copy()
        
        while remaining:
            # Find sub-queries with no unfulfilled dependencies
            ready = []
            for sq in remaining:
                dependencies_met = all(
                    any(completed.id == dep_id for completed in result)
                    for dep_id in sq.dependencies
                ) if sq.dependencies else True
                
                if dependencies_met:
                    ready.append(sq)
            
            if not ready:
                # Break circular dependencies by order
                ready = [min(remaining, key=lambda x: x.order)]
            
            # Sort ready sub-queries by order
            ready.sort(key=lambda x: x.order)
            result.extend(ready)
            
            for sq in ready:
                remaining.remove(sq)
        
        return result
    
    async def _mock_execute_query(self, query: Query, sub_queries: List[SubQuery],
                                context: ExecutionContext) -> QueryProcessingResult:
        """Mock execution when LangChain is not available."""
        start_time = time.time()
        
        # Simple mock responses based on query type
        mock_responses = {
            "comparative": f"Mock comparison analysis for: {query.original_text}",
            "analytical": f"Mock analytical response for: {query.original_text}",
            "explanatory": f"Mock explanation for: {query.original_text}",
            "summarization": f"Mock summary for: {query.original_text}",
            "factual": f"Mock factual answer for: {query.original_text}",
            "unknown": f"Mock response for: {query.original_text}"
        }
        
        final_answer = mock_responses.get(query.query_type.value, mock_responses["unknown"])
        
        # Create mock reasoning steps
        reasoning_steps = [
            {
                'id': 'mock-step-1',
                'description': 'Mock query analysis',
                'order': 1,
                'action': 'analyze',
                'result': 'Query analyzed successfully',
                'execution_time_ms': 100.0
            }
        ]
        
        return QueryProcessingResult(
            query_id=query.id,
            original_query=query.original_text,
            final_answer=final_answer,
            confidence=0.5,  # Lower confidence for mock
            reasoning_steps=reasoning_steps,
            sub_queries=[asdict(sq) for sq in sub_queries],
            execution_time_ms=(time.time() - start_time) * 1000,
            metadata={'mock_execution': True}
        )
    
    # Tool implementation methods
    def _search_information(self, query: str) -> str:
        """Mock search tool implementation."""
        return f"Search results for: {query}\n[Mock information retrieved]"
    
    def _analyze_data(self, data: str) -> str:
        """Mock analysis tool implementation."""
        return f"Analysis of provided data:\n[Mock analysis results]"
    
    def _compare_entities(self, entities: str) -> str:
        """Mock comparison tool implementation."""
        return f"Comparison of {entities}:\n[Mock comparison results]"
    
    def _summarize_information(self, information: str) -> str:
        """Mock summarization tool implementation."""
        return f"Summary of information:\n[Mock summary]"
