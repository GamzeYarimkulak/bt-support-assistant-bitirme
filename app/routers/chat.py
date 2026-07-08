"""
Chat endpoint for user questions and RAG-based answering.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import structlog
from datetime import datetime
import os
import json
import csv
from pathlib import Path

from app.config import settings
from core.rag.pipeline import RAGPipeline, RAGResult
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.embedding_retriever import EmbeddingRetriever
from data_pipeline.build_indexes import IndexBuilder

router = APIRouter()
logger = structlog.get_logger()

# Global RAG pipeline instance (initialized lazily)
_rag_pipeline: Optional[RAGPipeline] = None

# Conversation memory store (session_id -> list of messages)
# Each message: {"role": "user"|"assistant", "content": str, "timestamp": datetime}
_conversation_memory: Dict[str, List[Dict[str, Any]]] = {}
MAX_MEMORY_MESSAGES = 10  # Last 5 user + 5 assistant messages (5 pairs)


class ChatMessage(BaseModel):
    """Conversation message supplied by the browser for context recovery."""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., min_length=1, max_length=4000, description="Message content")


class ChatRequest(BaseModel):
    """User chat request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")
    language: Optional[str] = Field(None, description="Query language (auto-detected if not provided)")
    messages: Optional[List[ChatMessage]] = Field(
        None,
        description="Optional recent conversation history from the browser"
    )


class Source(BaseModel):
    """Source document used for answering."""
    doc_id: str
    doc_type: str  # "ticket" or "document"
    title: str
    snippet: str
    relevance_score: float


class RetrievalDebugInfo(BaseModel):
    """Debug information about retrieval process."""
    alpha_used: Optional[float] = None
    bm25_results_count: Optional[int] = None
    embedding_results_count: Optional[int] = None
    hybrid_results_count: Optional[int] = None
    query_type: Optional[str] = None  # "short", "medium", "long", "technical"


class ChatResponse(BaseModel):
    """Chat response with answer and sources."""
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[Source]
    has_answer: bool
    intent: Optional[str] = None
    language: str
    timestamp: datetime
    debug_info: Optional[RetrievalDebugInfo] = None  # Debug info for retrieval process


def get_conversation_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of messages (oldest to newest)
    """
    return _conversation_memory.get(session_id, [])


def normalize_request_history(messages: Optional[List[ChatMessage]]) -> List[Dict[str, Any]]:
    """Convert browser-supplied messages into the internal memory format."""
    normalized: List[Dict[str, Any]] = []
    for message in messages or []:
        role = message.role.strip().lower()
        content = message.content.strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    return normalized[-MAX_MEMORY_MESSAGES:]


def replace_conversation_history(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """Replace server memory with a trusted recent browser history snapshot."""
    _conversation_memory[session_id] = messages[-MAX_MEMORY_MESSAGES:]


def add_to_conversation_history(session_id: str, role: str, content: str) -> None:
    """
    Add a message to conversation history.
    
    Args:
        session_id: Session identifier
        role: "user" or "assistant"
        content: Message content
        
    Note:
        Automatically trims history to MAX_MEMORY_MESSAGES
    """
    if session_id not in _conversation_memory:
        _conversation_memory[session_id] = []
    
    _conversation_memory[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Trim to keep only last N messages
    if len(_conversation_memory[session_id]) > MAX_MEMORY_MESSAGES:
        _conversation_memory[session_id] = _conversation_memory[session_id][-MAX_MEMORY_MESSAGES:]
    
    try:
        logger.debug("conversation_updated", 
                    session_id=session_id, 
                    role=role,
                    total_messages=len(_conversation_memory[session_id]))
    except (OSError, UnicodeError):
        pass


def _save_chat_log(
    request: ChatRequest,
    response: ChatResponse,
    rag_result: RAGResult
) -> None:
    """
    Save chat interaction to log files.
    
    Saves:
    1. JSONL log file with full conversation details
    2. CSV ticket file (if save_as_tickets enabled) for anomaly detection
    
    Args:
        request: Chat request
        response: Chat response
        rag_result: RAG result with metadata
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.chat_logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique ticket ID
    ticket_id = f"CHAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.session_id or 'unknown'}"
    
    # Prepare log entry
    log_entry = {
        "ticket_id": ticket_id,
        "session_id": request.session_id,
        "timestamp": datetime.now().isoformat(),
        "language": response.language,
        "user_query": request.query,
        "assistant_answer": response.answer,
        "confidence": response.confidence,
        "has_answer": response.has_answer,
        "intent": response.intent,
        "num_sources": len(response.sources),
        "sources": [
            {
                "doc_id": src.doc_id,
                "doc_type": src.doc_type,
                "title": src.title,
                "relevance_score": src.relevance_score
            }
            for src in response.sources
        ],
        "debug_info": response.debug_info.dict() if response.debug_info else None
    }
    
    # Save to JSONL file (append mode)
    jsonl_file = log_dir / "chat_logs.jsonl"
    with open(jsonl_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    # Save as ticket CSV (if enabled) for anomaly detection
    if settings.save_as_tickets:
        csv_file = log_dir / "chat_tickets.csv"
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    "ticket_id", "created_at", "short_description", "description",
                    "category", "subcategory", "priority", "status", "resolution",
                    "language", "confidence", "has_answer", "session_id"
                ])
            
            # Determine category from query (simple heuristic)
            category = "General"
            query_lower = request.query.lower()
            if any(kw in query_lower for kw in ["outlook", "email", "mail"]):
                category = "Email"
            elif any(kw in query_lower for kw in ["vpn", "bağlantı", "connection"]):
                category = "Network"
            elif any(kw in query_lower for kw in ["şifre", "password", "login"]):
                category = "Security"
            elif any(kw in query_lower for kw in ["yazıcı", "printer"]):
                category = "Hardware"
            elif any(kw in query_lower for kw in ["windows", "sistem", "system"]):
                category = "System"
            
            # Determine priority based on confidence
            priority = "Low"
            if response.confidence >= 0.8:
                priority = "High"
            elif response.confidence >= 0.6:
                priority = "Medium"
            
            writer.writerow([
                ticket_id,
                datetime.now().isoformat(),
                request.query[:200],  # short_description
                response.answer[:1000] if response.has_answer else "No answer provided",  # description
                category,
                "",  # subcategory
                priority,
                "Closed" if response.has_answer else "Open",
                response.answer[:500] if response.has_answer else "",  # resolution
                response.language,
                response.confidence,
                "Yes" if response.has_answer else "No",
                request.session_id or ""
            ])
    
    logger.debug("chat_log_saved", ticket_id=ticket_id, log_dir=str(log_dir))


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or initialize the global RAG pipeline instance.
    
    This function lazily loads the indexes and creates the RAG pipeline.
    In production, this would be initialized at startup.
    
    Returns:
        RAGPipeline instance
        
    Raises:
        HTTPException: If indexes cannot be loaded
    """
    global _rag_pipeline
    
    if _rag_pipeline is not None:
        return _rag_pipeline
    
    try:
        # Load indexes from disk
        index_builder = IndexBuilder(index_dir="indexes/")
        
        bm25_retriever = index_builder.load_bm25_index()
        embedding_retriever = index_builder.load_embedding_index()
        
        if not bm25_retriever or not embedding_retriever:
            error_msg = "Indexes not found or failed to load"
            print(f"[ERROR] {error_msg}: bm25={bm25_retriever is not None}, embedding={embedding_retriever is not None}")
            raise HTTPException(status_code=503, detail=error_msg)
        
        # Create hybrid retriever with dynamic weighting enabled
        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            embedding_retriever=embedding_retriever,
            alpha=0.5,  # Default/fallback alpha
            use_dynamic_weighting=True,  # Enable dynamic alpha computation based on query
            kb_boost_enabled=settings.kb_boost_enabled,  # Enable KB document boosting
            kb_boost_factor=settings.kb_boost_factor  # KB boost factor
        )
        
        # Create RAG pipeline (PHASE 8: with real LLM support)
        _rag_pipeline = RAGPipeline(
            retriever=hybrid_retriever,
            confidence_threshold=0.6,  # Adjust based on testing
            # PHASE 8: Real LLM settings from config
            use_real_llm=settings.use_real_llm,
            openai_api_key=settings.openai_api_key,
            llm_model_name=settings.llm_model,
            llm_temperature=settings.llm_temperature,
            llm_max_tokens=settings.llm_max_tokens
        )
        
        try:
            logger.info("rag_pipeline_initialized_successfully",
                       use_real_llm=settings.use_real_llm,
                       llm_model=settings.llm_model if settings.use_real_llm else "stub")
        except (OSError, UnicodeError):
            pass
        return _rag_pipeline
        
    except Exception as e:
        try:
            logger.error("rag_pipeline_initialization_failed", error=str(e))
        except (OSError, UnicodeError):
            pass
        raise HTTPException(
            status_code=503,
            detail=f"RAG pipeline not available. Please build indexes first: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer user questions using hybrid RAG (PHASE 4 implementation).
    
    Args:
        request: User query and optional context
        
    Returns:
        Answer with sources and confidence score
        
    Process:
        1. Get or initialize RAG pipeline
        2. Call pipeline.answer() with the question
        3. Map RAGResult to ChatResponse
        4. Return structured response
    """
    try:
        # Log with safe encoding (avoid PowerShell Unicode issues)
        try:
            logger.info("chat_request", 
                       query=request.query[:50],  # Truncate to avoid encoding issues
                       session_id=request.session_id,
                       language=request.language)
        except (OSError, UnicodeError):
            pass  # Skip logging if encoding fails
        
        # Get RAG pipeline (initializes if needed)
        pipeline = get_rag_pipeline()
        
        # Get conversation history if session_id provided. Prefer the browser
        # snapshot when available so context survives page reloads/server restarts.
        conversation_history = []
        request_history = normalize_request_history(request.messages)
        if request_history:
            conversation_history = request_history
        elif request.session_id:
            conversation_history = get_conversation_history(request.session_id)
        if request.session_id:
            try:
                logger.info("conversation_history_loaded",
                           session_id=request.session_id,
                           history_length=len(conversation_history),
                           source="browser" if request_history else "server")
            except (OSError, UnicodeError):
                pass
        
        # Call RAG pipeline with conversation context
        rag_result: RAGResult = pipeline.answer(
            question=request.query,
            language=request.language,
            session_id=request.session_id,
            conversation_history=conversation_history  # PHASE 9: Add conversation context
        )
        
        # Convert sources from RAGResult to ChatResponse format
        sources = [
            Source(
                doc_id=src["doc_id"],
                doc_type=src["doc_type"],
                title=src["title"],
                snippet=src["snippet"],
                relevance_score=src["relevance_score"]
            )
            for src in rag_result.sources
        ]
        
        # Build debug info
        debug_info = None
        if rag_result.debug_info:
            debug_info = RetrievalDebugInfo(
                alpha_used=rag_result.debug_info.get("alpha_used"),
                bm25_results_count=rag_result.debug_info.get("bm25_results_count"),
                embedding_results_count=rag_result.debug_info.get("embedding_results_count"),
                hybrid_results_count=rag_result.debug_info.get("hybrid_results_count"),
                query_type=rag_result.debug_info.get("query_type")
            )
        
        # Build response
        response = ChatResponse(
            answer=rag_result.answer,
            confidence=rag_result.confidence,
            sources=sources,
            has_answer=rag_result.has_answer,
            intent=rag_result.intent,
            language=rag_result.language or "tr",
            timestamp=datetime.now(),
            debug_info=debug_info
        )
        
        try:
            logger.info("chat_response",
                       has_answer=response.has_answer,
                       confidence=response.confidence,
                       num_sources=len(sources))
        except (OSError, UnicodeError):
            pass
        
        # Save conversation to memory (PHASE 9: Conversation tracking)
        if request.session_id:
            if request_history:
                replace_conversation_history(request.session_id, request_history)
            add_to_conversation_history(request.session_id, "user", request.query)
            add_to_conversation_history(request.session_id, "assistant", rag_result.answer)
        
        # Save chat logs to file (if enabled)
        if settings.save_chat_logs:
            try:
                _save_chat_log(request, response, rag_result)
            except Exception as e:
                logger.warning("failed_to_save_chat_log", error=str(e))
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 503 from get_rag_pipeline)
        raise
    except Exception as e:
        try:
            logger.error("chat_error", error=str(e), query=request.query[:50])
        except (OSError, UnicodeError):
            pass
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


