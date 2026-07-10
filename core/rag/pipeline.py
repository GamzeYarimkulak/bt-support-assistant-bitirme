"""
Main RAG pipeline orchestrating retrieval and generation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import csv
import structlog
import os
import re

# OpenAI import - only needed if using real LLM
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # Placeholder

from core.retrieval.hybrid_retriever import HybridRetriever
from core.rag.prompts import PromptBuilder
from core.rag.confidence import ConfidenceEstimator
from core.nlp.it_relevance import ITRelevanceChecker

logger = structlog.get_logger()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RAGResult:
    """
    Result from RAG pipeline containing answer and metadata.
    
    This structure is returned by the RAG pipeline and contains:
    - The generated answer (or "no answer" message)
    - Confidence score
    - Source documents used
    - Whether a reliable answer was generated
    - Optional language and intent information
    - Optional debug information about retrieval process
    """
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    has_answer: bool
    language: Optional[str] = None
    intent: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None  # Debug info: alpha_used, query_type, etc.


# ============================================================================
# Real LLM Function (PHASE 8 - OpenAI Integration)
# ============================================================================

def generate_answer_with_llm(
    question: str, 
    docs: List[Dict[str, Any]], 
    language: str = "tr",
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1500
) -> str:
    """
    Generate advisory-style answers using real LLM (OpenAI GPT).
    
    This function maintains the same ADVISORY behavior as the stub:
    - Presents retrieved information as PAST EXAMPLES
    - Uses recommendation language
    - NEVER claims to have performed actions for the user
    
    PHASE 9: Now supports conversation history for context-aware answers!
    
    Args:
        question: User's question in Turkish or English
        docs: Retrieved documents (tickets + PDFs)
        language: Response language ('tr' or 'en')
        conversation_history: Previous conversation messages (PHASE 9)
        api_key: OpenAI API key
        model: OpenAI model name (gpt-4o-mini, gpt-4o, etc.)
        temperature: Creativity (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum response length
    
    Returns:
        Advisory-style answer in the requested language
    """
    if not docs:
        if language == "tr":
            return "Üzgünüm, bu konuda yeterli bilgi bulunamadı. Lütfen sorunuzu farklı kelimelerle tekrar deneyin veya BT destek ekibiyle iletişime geçin."
        return "I'm sorry, I couldn't find sufficient information on this topic. Please try rephrasing your question or contact the IT support team."

    direct_kb_docs = _direct_kb_docs_for_question(question, docs)
    if direct_kb_docs and _extract_relevant_kb_segments(question, direct_kb_docs):
        if language == "tr":
            return _build_direct_kb_answer_tr(question, direct_kb_docs)
        return _build_direct_kb_answer_en(question, direct_kb_docs)

    # Check if OpenAI package is available
    if not OPENAI_AVAILABLE:
        logger.warning("openai_package_not_installed_using_stub",
                      message="openai package not installed, falling back to stub")
        return generate_answer_with_stub(question, docs, language)
    
    if not api_key:
        logger.warning("no_api_key_using_stub", 
                      api_key_provided=api_key is not None,
                      api_key_value=f"{api_key[:10]}..." if api_key else "None")
        return generate_answer_with_stub(question, docs, language)
    
    logger.info("real_llm_call_initiated",
               api_key_length=len(api_key) if api_key else 0,
               model=model,
               language=language)

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Build context from retrieved documents
        context = _build_context_for_llm(docs, language)
        
        # Build system and user prompts
        system_prompt = _build_system_prompt(language)
        user_prompt = _build_user_prompt(question, context, language)
        
        logger.info(
            "calling_openai_api",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            language=language,
            conversation_history_length=len(conversation_history or [])
        )
        
        # Build messages with conversation history (PHASE 9)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add previous conversation if available
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current question with context
        messages.append({"role": "user", "content": user_prompt})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        
        logger.info(
            "openai_response_received",
            tokens_used=response.usage.total_tokens,
            answer_length=len(answer)
        )
        
        return answer
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("openai_api_error", 
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=error_details)
        # Fallback to stub on error
        return generate_answer_with_stub(question, docs, language)


def _build_context_for_llm(docs: List[Dict[str, Any]], language: str) -> str:
    """Build formatted context from retrieved documents for LLM."""
    context_parts = []
    
    for i, doc in enumerate(docs[:5], 1):  # Top 5 documents
        doc_type = doc.get("doc_type") or doc.get("type", "ticket")
        
        if _is_kb_document(doc):
            title = doc.get("title", "Dokümantasyon")
            content = doc.get("content") or doc.get("text", "")
            context_parts.append(f"[DÖKÜMAN {i}] {title}\n{content[:1500]}")
        else:
            ticket_id = doc.get("ticket_id", f"TCK-{i:04d}")
            issue = doc.get("issue_description") or doc.get("short_description", "")
            resolution = doc.get("resolution", "")
            context_parts.append(f"[TICKET {i}] ID: {ticket_id}\nSorun: {issue}\nÇözüm: {resolution[:800]}")
    
    return "\n\n".join(context_parts)


def _build_system_prompt(language: str) -> str:
    """Build system prompt that enforces advisory behavior and step-by-step guidance."""
    if language == "tr":
        return """Sen bir BT destek asistanısın. Görevin, kullanıcılara GEÇMİŞ ÇÖZÜM ÖRNEKLERİNE dayalı ÖNERİLER sunmaktır.

**KRİTİK KURAL:**
- ASLA kullanıcı için bir işlem yaptığını iddia etme
- "Şifreniz sıfırlandı", "Dosyanızı gönderdim" gibi ifadeler YASAK
- Bunun yerine "BT ekibi genellikle şu adımları uygular", "Bu adımları deneyebilirsiniz" kullan

**Yanıt Formatı (ZORUNLU):**
1. Sorunu kısa bir cümleyle özetle
2. Çözüm adımlarını şu formatta yaz:

**Adım 1: [Başlık]**
   - Alt adım veya açıklama
   - Nereye tıklayacağını belirt
   
**Adım 2: [Başlık]**
   - Alt adım
   - Detaylı açıklama

3. Her adımı KISA VE NET tut (maksimum 2-3 cümle)
4. Alt adımlar için tire (-) veya bullet (•) kullan
5. Önemli kelimeler için **bold** kullan
6. Adımlar arası boşluk bırak
7. Sonunda: "Bu adımları kendiniz deneyebilir veya BT ekibinden destek isteyebilirsiniz."

**TAKİP SORULARI:**
- Eğer kullanıcı belirsiz bir takip sorusu sorarsa (örn: "nereden resetleyebilirim?", "diğer adımlarda ne yapacaktım"):
  1. Önceki konuşma geçmişine bakarak TAHMİN ET
  2. En olası çözümü sun
  3. Alternatif ihtimalleri de GÖSTER (örn: "VPN resetinden mi bahsediyorsunuz? Yoksa şifre sıfırlama mı?")
  4. Eğer kullanıcı "diğer adımlar" veya "sonraki adımlar" diye sorarsa, önceki mesajlarda verdiğiniz adımları hatırlatın

**TEŞEKKÜR MESAJLARI:**
- Eğer kullanıcı "tamamdır", "teşekkür ederim", "tamam teşekkür" gibi mesajlar gönderirse:
  1. Kısa ve nazik bir yanıt verin (örn: "Rica ederim, başka bir konuda yardımcı olabilir miyim?")
  2. Önceki konuşmada bir sorun varsa, o sorunla ilgili kısa bir özet sunun
  3. Uzun açıklamalar yapmayın, sadece nezaket gösterin

**Ton:** Profesyonel, yardımcı, önerici (emredici değil)"""
    else:
        return """You are an IT support assistant. Your role is to provide RECOMMENDATIONS based on PAST SOLUTION EXAMPLES.

**CRITICAL RULE:**
- NEVER claim you have performed an action for the user
- "Your password has been reset", "I sent your file" are FORBIDDEN
- Instead use "The IT team typically applies these steps", "You can try these steps"

**Response Format (MANDATORY):**
1. Summarize the issue in one short sentence
2. Write solution steps in this format:

**Step 1: [Title]**
   - Sub-step or explanation
   - Specify where to click
   
**Step 2: [Title]**
   - Sub-step
   - Detailed explanation

3. Keep each step SHORT and CLEAR (maximum 2-3 sentences)
4. Use dashes (-) or bullets (•) for sub-steps
5. Use **bold** for important words
6. Add blank lines between steps
7. End with: "You can try these steps yourself or request support from IT team."

**FOLLOW-UP QUESTIONS:**
- If the user asks a vague follow-up question (e.g., "where can I reset it?"):
  1. INFER from previous conversation history
  2. Provide the most likely solution
  3. Also SHOW alternative possibilities (e.g., "Are you referring to VPN reset? Or password reset?")

**Tone:** Professional, helpful, advisory (not commanding)"""


def _build_user_prompt(question: str, context: str, language: str) -> str:
    """Build user prompt with question and context."""
    if language == "tr":
        return f"""Kullanıcı Sorusu: {question}

Benzer Durumlardan Örnekler:
{context}

Yukarıdaki örneklere dayanarak, kullanıcıya ADIM ADIM, okunaklı ve uygulanabilir öneriler sun.

ÖRNEK FORMAT:
**Adım 1: VPN Ayarlarını Açın**
- **Başlat** menüsünden **Ayarlar**'ı seçin
- **Ağ ve İnternet** > **VPN** sekmesine gidin

**Adım 2: Bağlantıyı Sıfırlayın**
- Mevcut VPN bağlantısının yanındaki **"..."** butonuna tıklayın
- **Bağlantıyı Sil** ve **Yeniden Ekle** seçeneğini kullanın

Bu formatta, kısa ve net adımlarla cevap ver."""
    else:
        return f"""User Question: {question}

Examples from Similar Cases:
{context}

Based on the examples above, provide STEP-BY-STEP, readable, and actionable recommendations.

EXAMPLE FORMAT:
**Step 1: Open VPN Settings**
- Select **Settings** from **Start** menu
- Go to **Network & Internet** > **VPN** tab

**Step 2: Reset Connection**
- Click the **"..."** button next to your VPN connection
- Use **Delete** and **Re-add** options

Answer in this format with short and clear steps."""


# ============================================================================
# LLM Stub Function (PHASE 6.5 - Advisory/Recommendation Style)
# ============================================================================

def generate_answer_with_stub(question: str, docs: List[Dict[str, Any]], language: str = "tr") -> str:
    """
    Advisory-style answer generation stub (PHASE 6.5).
    
    IMPORTANT BEHAVIOR:
    This function acts as a RECOMMENDATION / DECISION SUPPORT system, NOT an agent
    that performs actions. It presents past ticket resolutions as EXAMPLES of what
    IT teams have done in similar situations.
    
    CRITICAL SAFETY RULE:
    The assistant MUST NOT claim that it has already performed any action for the user.
    For example:
    - ❌ WRONG: "Şifreniz sıfırlandı" (Your password has been reset)
    - ❌ WRONG: "Bağlantınızı gönderdim" (I sent your link)
    - ✅ CORRECT: "BT ekibi genellikle şifre sıfırlama bağlantısı gönderir"
    - ✅ CORRECT: "Bu adımları denemeniz önerilir"
    
    The answer should:
    1. Present retrieved ticket resolutions as past examples
    2. Use advisory language ("önerilir", "deneyebilirsiniz", "BT ekibi genellikle...")
    3. Suggest that the user can try these steps OR request them from IT support
    4. NEVER claim actions were already performed for THIS user
    
    FUTURE INTEGRATION POINT:
    Replace this function with a real LLM call that follows the same advisory principles.
    
    Args:
        question: User's question
        docs: List of retrieved documents (past ITSM tickets)
        language: Language code ("tr" for Turkish, "en" for English)
        
    Returns:
        Advisory-style answer string
        
    Example:
        >>> docs = [{"short_description": "Outlook şifremi unuttum", 
        ...          "resolution": "Şifre sıfırlama bağlantısı gönderildi"}]
        >>> answer = generate_answer_with_stub("Outlook şifremi unuttum", docs)
        >>> # Returns advisory answer, NOT "Şifreniz sıfırlandı"
    """
    if not docs:
        if language == "tr":
            return "Mevcut kaynaklara dayanarak güvenilir bir cevap üretemiyorum."
        else:
            return "I cannot provide a reliable answer based on available sources."

    direct_kb_docs = _direct_kb_docs_for_question(question, docs)
    if direct_kb_docs and _extract_relevant_kb_segments(question, direct_kb_docs):
        if language == "tr":
            return _build_direct_kb_answer_tr(question, direct_kb_docs)
        return _build_direct_kb_answer_en(question, direct_kb_docs)
    
    # Build advisory-style answer using examples from past tickets
    if language == "tr":
        return _build_advisory_answer_tr(question, docs)
    else:
        return _build_advisory_answer_en(question, docs)


def _doc_identifier(doc: Dict[str, Any]) -> str:
    """Return a readable source identifier across ticket and KB documents."""
    for field in ("ticket_id", "doc_id", "id", "document_id"):
        value = doc.get(field)
        if value:
            return str(value)
    return "Bilinmeyen"


def _doc_title(doc: Dict[str, Any]) -> str:
    """Return the best available issue/title field for a retrieved document."""
    for field in ("short_description", "title", "subject", "category"):
        value = str(doc.get(field, "") or "").strip()
        if value:
            return value

    text = str(doc.get("text", "") or doc.get("description", "") or "").strip()
    return text[:120]


def _doc_solution_text(doc: Dict[str, Any]) -> str:
    """Return usable answer context even when a KB chunk has no resolution field."""
    for field in ("resolution", "answer", "solution", "content", "text", "description"):
        value = str(doc.get(field, "") or "").strip()
        if value:
            return value
    return ""


def _doc_document_id(doc: Dict[str, Any]) -> str:
    """Return the parent KB document id for chunk-level search results."""
    for field in ("document_id", "source_document_id"):
        value = str(doc.get(field, "") or "").strip()
        if value:
            return value

    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        value = str(metadata.get("document_id", "") or "").strip()
        if value:
            return value

    identifier = str(doc.get("doc_id", "") or doc.get("id", "") or "").strip()
    if "_chunk_" in identifier:
        return identifier.split("_chunk_", 1)[0]
    return ""


@lru_cache(maxsize=1)
def _processed_kb_documents_by_id() -> Dict[str, str]:
    """Load full processed KB document text for source-grounded chunk answers."""
    path = PROJECT_ROOT / "data" / "processed" / "kb_documents.csv"
    if not path.exists():
        return {}

    documents: Dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                document_id = str(row.get("document_id", "") or "").strip()
                content = str(row.get("content", "") or "").strip()
                if document_id and content:
                    documents[document_id] = content
    except Exception as exc:
        logger.warning("processed_kb_documents_load_failed", path=str(path), error=str(exc))
        return {}

    return documents


def _full_kb_source_text(doc: Dict[str, Any]) -> str:
    """Prefer full processed KB document text when a retrieved result is a chunk."""
    if _is_kb_document(doc):
        document_id = _doc_document_id(doc)
        if document_id:
            full_text = _processed_kb_documents_by_id().get(document_id, "").strip()
            if full_text:
                return full_text
    return _doc_solution_text(doc)


def _source_label_tr(doc_type: str) -> str:
    normalized = str(doc_type or "").casefold()
    if normalized in {"kb", "document", "pdf"}:
        return "Bilgi dokümanı"
    if normalized == "playbook":
        return "Genel BT kontrol listesi"
    return "Ticket"


def _source_label_en(doc_type: str) -> str:
    normalized = str(doc_type or "").casefold()
    if normalized in {"kb", "document", "pdf"}:
        return "Knowledge document"
    if normalized == "playbook":
        return "General IT checklist"
    return "Ticket"


def _is_kb_document(doc: Dict[str, Any]) -> bool:
    doc_type = str(doc.get("doc_type") or doc.get("type") or "").casefold()
    return doc_type in {"kb", "document", "pdf"} or str(doc.get("id", "")).startswith("ozdilek_kb_")


def _should_use_direct_kb_answer(question: str, docs: List[Dict[str, Any]]) -> bool:
    """Use direct source-grounded answers for factual KB/document questions."""
    return bool(_direct_kb_docs_for_question(question, docs))


def _is_factual_kb_question(question: str) -> bool:
    """Return True when the user asks for a factual value from documents."""
    normalized = question.casefold()
    ascii_normalized = _normalize_for_match(question)

    factual_markers = (
        "hangi",
        "nedir",
        "ne ",
        "kim",
        "kaç",
        "kac",
        "nerede",
        "nereden",
        "kullanılır",
        "kullanilir",
        "belirtil",
        "söyler",
        "soyler",
        "yazar",
        "hakkında",
        "hakkinda",
        "ürün",
        "urun",
        "olacak",
        "nasıl",
        "nasil",
        "yapılıyor",
        "yapiliyor",
        "bahseder",
        "bakar mısın",
        "bakar misin",
    )
    return any(marker in normalized or marker in ascii_normalized for marker in factual_markers)


def _requires_source_grounded_kb_answer(question: str) -> bool:
    """Identify document lookup questions that should not be answered creatively."""
    if not _is_factual_kb_question(question):
        return False

    normalized = question.casefold()
    ascii_normalized = _normalize_for_match(question)
    source_markers = (
        "özdilek",
        "ozdilek",
        "strateji",
        "talimat",
        "yönerge",
        "yonerge",
        "prosedür",
        "prosedur",
        "doküman",
        "dokuman",
        "belge",
        "bölüm",
        "bolum",
        "kısım",
        "kisim",
        "ürün",
        "urun",
        "kullanılıyor",
        "kullaniliyor",
        "kullanılır",
        "kullanilir",
        "olacakmış",
        "olacakmis",
        "aksiyon planı",
        "aksiyon plani",
        "ilke",
        "ilkeleri",
        "tasarim",
    )
    return any(marker in normalized or marker in ascii_normalized for marker in source_markers)


def _direct_kb_docs_for_question(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select KB/document candidates for direct factual answering."""
    if not docs or not _is_factual_kb_question(question):
        return []

    kb_docs = [doc for doc in docs[:5] if _is_kb_document(doc)]
    if not kb_docs:
        return []

    if _requires_source_grounded_kb_answer(question) or _is_kb_document(docs[0]):
        return kb_docs
    return []


_KB_STOPWORDS = {
    "acaba", "ama", "bana", "bende", "benim", "bir", "bunu", "hangi", "hakkinda",
    "hakkında", "icin", "için", "kismi", "kısmı", "kisminda", "kısmında", "mi",
    "ne", "nedir", "olan", "olarak", "soru", "sordum", "su", "şu", "ve", "veya",
    "ver", "bilgi", "bilgisi", "özdilek", "ozdilek", "peki", "bakar", "misin",
    "mısın", "teknoloji", "teknolojileri", "strateji", "stratejisi", "donanimsal",
    "donanımsal", "donanim", "donanım", "bolum", "bölüm", "bolumu", "bölümü",
}


_TURKISH_CHAR_TRANSLATION = str.maketrans(
    "\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc",
    "cgiosuCGIOSU",
)


def _normalize_for_match(text: str) -> str:
    normalized = str(text or "").translate(_TURKISH_CHAR_TRANSLATION).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _squash_elongated_letters(text: str) -> str:
    return re.sub(r"([a-z])\1{2,}", r"\1", text)


_ACKNOWLEDGMENT_COMPACT_MESSAGES = {
    "anladim",
    "cokiyi",
    "elinesaglik",
    "eyvallah",
    "harika",
    "harikasin",
    "mukemmel",
    "mukemmelsin",
    "ok",
    "okay",
    "sagol",
    "super",
    "supersin",
    "tamam",
    "tamamdir",
    "tesekkur",
    "tesekkurler",
    "thanks",
}

_ACKNOWLEDGMENT_PATTERNS = (
    re.compile(r"^(cok )?(tesekkur|tesekkurler)( ederim| ediyorum| ederiz)?$"),
    re.compile(r"^(cok )?(sag ol|sagol|eyvallah)$"),
    re.compile(r"^(tamam|ok|okay|anladim|tamamdir)( tesekkur| thanks)?( ederim| ediyorum)?$"),
    re.compile(r"^(super|supersin|harika|harikasin|mukemmel|mukemmelsin|cok iyi|eline saglik)$"),
)


def _is_acknowledgment_message(message: str) -> bool:
    normalized = _squash_elongated_letters(_normalize_for_match(message))
    if not normalized:
        return False

    compact = normalized.replace(" ", "")
    if compact in _ACKNOWLEDGMENT_COMPACT_MESSAGES:
        return True

    return any(pattern.match(normalized) for pattern in _ACKNOWLEDGMENT_PATTERNS)


def _normalize_for_match_with_offsets(text: str) -> tuple[str, List[int]]:
    """Normalize text like _normalize_for_match while keeping original offsets."""
    normalized_chars: List[str] = []
    offsets: List[int] = []
    pending_space = False

    for original_index, char in enumerate(str(text or "")):
        for normalized_char in char.translate(_TURKISH_CHAR_TRANSLATION).casefold():
            if ("a" <= normalized_char <= "z") or normalized_char.isdigit():
                if pending_space and normalized_chars:
                    normalized_chars.append(" ")
                    offsets.append(original_index)
                pending_space = False
                normalized_chars.append(normalized_char)
                offsets.append(original_index)
            else:
                pending_space = bool(normalized_chars)

    return "".join(normalized_chars), offsets


def _repair_mojibake_text(text: str) -> str:
    """Repair common UTF-8 text that was decoded as Latin-1 before matching."""
    value = str(text or "")
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value
    return repaired or value


def _normalize_for_scenario_match(text: str) -> str:
    value = str(text or "")
    repaired = _repair_mojibake_text(value)
    if repaired == value:
        return _normalize_for_match(value)
    return _normalize_for_match(f"{value} {repaired}")


def _query_terms(question: str) -> set[str]:
    terms: set[str] = set()
    suffixes = (
        "lerinde", "larinda", "lerinde", "lerinde", "lerinden", "larindan",
        "lerinden", "larindan", "inden", "indan", "ından", "lerden", "lardan",
        "sinde", "sinda", "daki", "deki", "daki", "den", "dan", "leri",
        "lari", "ları", "leri", "lar", "ler", "dir", "tir", "nin", "nın",
        "in", "un", "de", "da",
    )
    for token in _normalize_for_match(question).split():
        if len(token) < 3 or token in _KB_STOPWORDS:
            continue
        terms.add(token)
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                terms.add(token[: -len(suffix)])
                break
    return terms


def _split_kb_segments(text: str) -> List[str]:
    clean_text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean_text:
        return []

    numbered_segments = re.split(r"(?<![\w.])(?=[1-9](?:\.\d+){1,5}\.?\s*)", clean_text)
    segments: List[str] = []
    for segment in numbered_segments:
        segment = segment.strip(" -")
        if not segment:
            continue
        if len(segment) > 450:
            segments.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", segment) if part.strip())
        else:
            segments.append(segment)
    return segments


def _strip_inline_pdf_metadata(text: str) -> str:
    """Remove page header/footer fragments that land in the middle of OCR text."""
    cleaned = re.sub(
        r"\[Page\s+\d+\].{0,650}\bNO\s+\d+\s*",
        " ",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" -")


def _clean_kb_answer_segment(segment: str) -> str:
    """Remove common PDF/form headers before showing a KB sentence to users."""
    text = re.sub(r"\s+", " ", str(segment or "")).strip(" -")
    if not text:
        return ""

    metadata_markers = (
        "DOKÜMAN",
        "DOKUMAN",
        "REVİZYON",
        "REVIZYON",
        "YÜRÜRLÜK",
        "YURURLUK",
        "HAZIRLAYAN",
        "ONAY",
        "KODU",
        "KOD NO",
    )
    prefix = _normalize_for_match(text[:360])
    has_metadata_prefix = any(_normalize_for_match(marker) in prefix for marker in metadata_markers)

    if has_metadata_prefix:
        page_section_match = re.search(r"\[Page\s+\d+\]\s*\d+\.?\s*", text, flags=re.IGNORECASE)
        if page_section_match:
            text = text[page_section_match.end():].strip()
        else:
            numbered_match = re.search(r"\b\d+\.\s*(?=[A-ZÇĞİÖŞÜa-zçğıöşü])", text)
            if numbered_match and numbered_match.start() > 20:
                text = text[numbered_match.end():].strip()

    text = re.sub(r"^\[Page\s+\d+\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text).strip()
    return _strip_inline_pdf_metadata(text)


def _is_action_plan_question(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return (
        "aksiyon" in normalized
        and "plan" in normalized
        or "enerji yonetimi" in normalized
    )


def _is_erp_question(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return "erp" in normalized or ("sap" in normalized and "sistem" in normalized)


def _is_director_responsibility_question(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return (
        "direktor" in normalized
        and ("gorev" in normalized or "soruml" in normalized or "ne yap" in normalized or "bahseder" in normalized)
    )


def _extract_between_labels(text: str, start_label: str, end_labels: tuple[str, ...]) -> str:
    start = text.find(start_label)
    if start < 0:
        return ""
    start += len(start_label)
    end_positions = [text.find(label, start) for label in end_labels if text.find(label, start) >= 0]
    end = min(end_positions) if end_positions else len(text)
    return re.sub(r"\s+", " ", text[start:end]).strip(" :-")


def _extract_action_plan_segments(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    text = _full_kb_source_text(doc)
    title = _doc_title(doc)
    doc_id = _doc_identifier(doc)

    fields = [
        (
            "Proje amacı/hedefi",
            _extract_between_labels(
                text,
                "Proje Amacı/Hedefi:",
                ("Proje Başlangıç Tarihi:", "Proje Tanımı:", "Proje Bütçesi"),
            ),
        ),
        (
            "Proje tanımı",
            _extract_between_labels(
                text,
                "Proje Tanımı:",
                ("Proje Bütçesi", "Gerçekleşen Maliyet:", "Proje Planlama"),
            ),
        ),
        (
            "Aksiyonlar",
            _extract_between_labels(
                text,
                "Proje Planlama Aksiyonları:",
                ("Proje Sonuçlarının Doğrulanması:", "Sonuçların Değerlendirmesi:"),
            ),
        ),
        (
            "Proje sonuçlarının doğrulanması",
            _extract_between_labels(
                text,
                "Proje Sonuçlarının Doğrulanması:",
                ("Sonuçların Değerlendirmesi:", "Sorumlular:"),
            ),
        ),
        (
            "Sonuçların değerlendirmesi",
            _extract_between_labels(
                text,
                "Sonuçların Değerlendirmesi:",
                ("Sorumlular:", "[Page 1]", "[Page 2]", "İletişim/Eğitim Planı:", "Proje iyileştirmelerinin sürdürülmesi:"),
            ),
        ),
    ]

    segments: List[Dict[str, str]] = []
    for label, value in fields:
        if not value:
            continue
        segments.append(
            {
                "segment": f"{label}: {value}",
                "title": title,
                "doc_id": doc_id,
            }
        )
    return segments


def _extract_normalized_span(
    text: str,
    start_markers: tuple[str, ...],
    end_markers: tuple[str, ...],
) -> str:
    normalized, offsets = _normalize_for_match_with_offsets(text)
    if not normalized or not offsets:
        return ""

    start_matches: List[tuple[int, str]] = []
    for marker in start_markers:
        normalized_marker = _normalize_for_match(marker)
        marker_index = normalized.find(normalized_marker)
        if marker_index >= 0:
            start_matches.append((marker_index, normalized_marker))

    if not start_matches:
        return ""

    start_index, start_marker = min(start_matches, key=lambda item: item[0])
    content_start_index = min(start_index + len(start_marker), len(offsets) - 1)
    content_start = offsets[content_start_index]

    content_end_index = len(normalized)
    for marker in end_markers:
        normalized_marker = _normalize_for_match(marker)
        marker_index = normalized.find(normalized_marker, content_start_index)
        if marker_index >= 0:
            content_end_index = min(content_end_index, marker_index)

    content_end = offsets[content_end_index] if content_end_index < len(offsets) else len(text)
    return text[content_start:content_end].strip()


def _is_design_principles_question(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return "tasarim" in normalized and any(marker in normalized for marker in ("ilke", "ilkeleri"))


def _design_principles_target(question: str) -> str:
    if not _is_design_principles_question(question):
        return ""

    normalized = _normalize_for_match(question)
    if "gomulu" in normalized or ("yazilim" in normalized and "elektronik" not in normalized and "elekronik" not in normalized):
        return "embedded_software"
    if "elektronik" in normalized or "elekronik" in normalized:
        return "electronic"
    return ""


def _design_principles_config(target: str) -> Optional[Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {
        "electronic": {
            "start": ("elektronik tasarim ilkeleri", "elekronik tasarim ilkeleri"),
            "end": (
                "gomulu yazilim tasarim ilkeleri",
                "elektronik tasarim standartlari",
                "elekronik tasarim standartlari",
            ),
            "labels": (
                "modulerlik",
                "dokumantasyon",
                "tasarim dogrulama ve test",
                "guc yonetimi",
                "emi emc uyumlulugu",
                "termal yonetim",
                "isi yonetimi",
                "yapilabilirlik ve uretilebilirlik",
            ),
            "limit": 8,
        },
        "embedded_software": {
            "start": ("gomulu yazilim tasarim ilkeleri",),
            "end": (
                "elektronik tasarim standartlari",
                "elekronik tasarim standartlari",
                "gomulu yazilim standartlari",
                "4 sorumlular",
                "5 dagitim",
            ),
            "labels": (
                "verimlilik",
                "tasinabilirlik",
                "gercek zamanli isletim",
                "hata yonetimi ve guvenlik",
                "yazilim guncellemeleri",
                "modulerlik ve yeniden kullanilabilirlik",
                "dokumantasyon",
                "test ve dogrulama",
            ),
            "limit": 8,
        },
    }
    return configs.get(target)


def _extract_labeled_design_segments(
    block: str,
    labels: tuple[str, ...],
    title: str,
    doc_id: str,
    limit: int,
) -> List[Dict[str, str]]:
    clean_block = _strip_inline_pdf_metadata(block)
    normalized, offsets = _normalize_for_match_with_offsets(clean_block)
    if not normalized or not offsets:
        return []

    label_positions: List[tuple[int, str]] = []
    search_from = 0
    for label in labels:
        position = normalized.find(label, search_from)
        if position < 0:
            position = normalized.find(label)
        if position < 0:
            continue
        label_positions.append((position, label))
        search_from = position + len(label)

    label_positions.sort(key=lambda item: item[0])
    selected: List[Dict[str, str]] = []
    seen: set[str] = set()

    for index, (position, label) in enumerate(label_positions):
        label_start = offsets[position]
        label_end_index = min(position + len(label) - 1, len(offsets) - 1)
        label_end = offsets[label_end_index] + 1
        segment_end = offsets[label_positions[index + 1][0]] if index + 1 < len(label_positions) else len(clean_block)

        label_text = re.sub(r"\s+", " ", clean_block[label_start:label_end]).strip(" :-")
        body = re.sub(r"\s+", " ", clean_block[label_end:segment_end]).strip(" :-")
        if len(body) < 12:
            continue

        segment = f"{label_text}: {body}" if label_text else body
        if len(segment) > 700:
            segment = segment[:697].rstrip() + "..."

        key = _normalize_for_match(segment)[:180]
        if key in seen:
            continue
        seen.add(key)
        selected.append({"segment": segment, "title": title, "doc_id": doc_id})
        if len(selected) >= limit:
            break

    return selected


def _extract_design_principle_segments(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    target = _design_principles_target(question)
    config = _design_principles_config(target)
    if not config:
        return []

    for doc in docs[:5]:
        if not _is_kb_document(doc):
            continue

        source_text = _full_kb_source_text(doc)
        if not source_text:
            continue

        block = _extract_normalized_span(source_text, config["start"], config["end"])
        if not block:
            continue

        title = _doc_title(doc)
        doc_id = _doc_document_id(doc) or _doc_identifier(doc)
        segments = _extract_labeled_design_segments(
            block=block,
            labels=config["labels"],
            title=title,
            doc_id=doc_id,
            limit=config["limit"],
        )
        if segments:
            return segments

    return []


def _is_desktop_laptop_maintenance_question(question: str) -> bool:
    normalized = _normalize_for_match(question)
    has_device = (
        ("dizustu" in normalized or "laptop" in normalized)
        and ("masaustu" in normalized or "bilgisayar" in normalized)
    )
    has_maintenance_intent = any(
        marker in normalized
        for marker in ("bakim", "soruml", "sure", "aralik", "kim", "kac", "yilda")
    )
    return has_device and has_maintenance_intent


def _extract_desktop_laptop_maintenance_segments(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    device_label = "Dizüstü ve masaüstü bilgisayar ve aksamları"
    next_device_pattern = (
        r"Tüm yazıcılar|Tum yazicilar|Plotterlar|Adisyon|El Terminalleri|"
        r"Switch|Sunucular|POS Kasa|POS Sunucuları|POS Sunuculari|POS Kiosk|Firewall"
    )

    for doc in docs[:5]:
        if not _is_kb_document(doc):
            continue

        title = _doc_title(doc)
        doc_id = _doc_identifier(doc)
        source_text = re.sub(r"\s+", " ", _full_kb_source_text(doc)).strip()
        match = re.search(
            rf"({re.escape(device_label)})\s+(.*?)(?=\s+(?:{next_device_pattern})\b|$)",
            source_text,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        row_tail = match.group(2).strip(" :-")
        interval = ""
        interval_match = re.search(r"\b(\d+)\s*$", row_tail)
        if interval_match:
            interval = interval_match.group(1)
            row_tail = row_tail[: interval_match.start()].strip(" :-")

        responsible = row_tail
        maintainer = ""
        for marker in ("İç Kaynaklar", "Dış Kaynaklar", "Ic Kaynaklar", "Dis Kaynaklar"):
            idx = row_tail.find(marker)
            if idx >= 0:
                responsible = row_tail[:idx].strip(" :-")
                maintainer = row_tail[idx:].strip(" :-")
                break

        details = [f"{device_label}: sorumlu birim {responsible or row_tail}"]
        if maintainer:
            details.append(f"bakımı yapan {maintainer}")
        if interval:
            details.append(f"bakım aralığı yılda {interval}")

        return [
            {
                "segment": "; ".join(details) + ".",
                "title": title,
                "doc_id": doc_id,
            }
        ]

    return []


def _extract_erp_segments(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return SAP/ERP-related source snippets for Özdilek system questions."""
    markers = ("sap", "hana", " oep", " orp", "ecommerce cloud", "bw ")
    selected: List[Dict[str, str]] = []
    seen: set[str] = set()

    def summarize_sap_evidence(title: str, segment: str) -> str:
        surface = _normalize_for_match(f"{title} {segment}")
        components: List[str] = []
        for marker, label in (
            ("hana", "SAP HANA"),
            ("orp", "SAP ORP"),
            ("oep", "SAP OEP"),
            ("retail", "SAP Retail"),
            ("uretim", "SAP Üretim"),
            ("ecommerce cloud", "SAP ecommerce cloud"),
            ("bw", "BW raporlama"),
        ):
            if marker in surface and label not in components:
                components.append(label)

        if not components:
            components.append("SAP")

        excerpt = re.sub(r"\s+", " ", segment).strip()
        if len(excerpt) > 240:
            excerpt = excerpt[:237].rstrip() + "..."

        component_text = ", ".join(components)
        return (
            "Özdilek kaynaklarında ERP/SAP sistemi olarak SAP geçiyor. "
            f"İlgili kaynakta {component_text} ifadesi yer alıyor. "
            f"Kaynak ifadesi: {excerpt}"
        )

    for doc in docs[:5]:
        if not _is_kb_document(doc):
            continue

        title = _doc_title(doc)
        doc_id = _doc_identifier(doc)
        source_text = _doc_solution_text(doc)
        title_surface = _normalize_for_match(title)
        doc_surface = _normalize_for_match(f"{title} {source_text[:1200]}")

        if not any(marker.strip() in doc_surface for marker in markers):
            continue

        candidate_segments = _split_kb_segments(source_text) or [source_text]
        for segment in candidate_segments:
            segment_surface = _normalize_for_match(f"{title} {segment}")
            if not any(marker.strip() in segment_surface for marker in markers):
                continue

            key = _normalize_for_match(f"{title} {segment}")[:220]
            if key in seen:
                continue
            seen.add(key)

            label = "ERP/SAP kaynağı"
            if "hana" in title_surface or "hana" in segment_surface:
                label = "SAP HANA kaynağı"
            elif "oep" in title_surface or "orp" in title_surface:
                label = "SAP sistem kaynağı"

            selected.append(
                {
                    "segment": summarize_sap_evidence(title, segment),
                    "title": title,
                    "doc_id": doc_id,
                }
            )
            break

        if len(selected) >= 3:
            break

    return selected


def _extract_director_responsibility_segments(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract role/responsibility blocks for BT and Ar-Ge director questions."""
    role_pattern = re.compile(
        r"(?:Bilgi Teknolojileri ve Ar-?Ge Direktörü|BT ve Ar-?Ge Direktörü):?\s*"
        r"(?P<body>.*?)"
        r"(?=(?:İç Denetim Müdürü|İlgili BT Birim Yöneticisi|E-Commerce BT Yöneticisi|"
        r"Tedarik Zinciri Direktörlüğü|DOKÜMAN ONAY|DOKÜMAN SON|REVİZYON|5\.DAĞITIM|"
        r"\b\d+\.[A-ZÇĞİÖŞÜ]|$))",
        flags=re.IGNORECASE | re.DOTALL,
    )
    selected: List[Dict[str, str]] = []
    seen: set[str] = set()

    for doc in docs[:5]:
        if not _is_kb_document(doc):
            continue

        title = _doc_title(doc)
        doc_id = _doc_identifier(doc)
        source_text = _full_kb_source_text(doc)
        normalized_surface = _normalize_for_match(f"{title} {source_text[:1500]}")
        if "direktor" not in normalized_surface or "soruml" not in normalized_surface:
            continue

        for match in role_pattern.finditer(source_text):
            body = re.sub(r"\s+", " ", match.group("body")).strip(" :-")
            if len(body) < 25:
                continue
            if not any(marker in body for marker in ("•", "sorumludur", "BT Stratejik", "Alımlar", "gündem", "yönlendirmeyi")):
                continue

            key = _normalize_for_match(f"{title} {body}")[:220]
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "segment": f"Bilgi Teknolojileri ve Ar-Ge Direktörü görevleri: {body}",
                    "title": title,
                    "doc_id": doc_id,
                }
            )
            break

        if len(selected) >= 3:
            break

    return selected


def _is_product_name_question(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return "urun" in normalized or "hangi urun" in normalized or "hangi model" in normalized


def _has_specific_named_value(segment: str) -> bool:
    normalized = _normalize_for_match(segment)
    known_values = (
        "sap",
        "hana",
        "point mobile",
        "lukhan sewoo",
        "xplore",
        "cisco",
        "dell",
        "hp",
    )
    if any(value in normalized for value in known_values):
        return True
    return bool(re.search(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]+(?:\s+[A-Z0-9][A-Za-z0-9-]+){1,4}\s+\d", segment))


def _section_number_and_body(segment: str) -> tuple[str, str]:
    match = re.match(r"^(?P<number>[1-9](?:\.\d+){1,5})\.?\s*(?P<body>.*)$", str(segment or "").strip())
    if not match:
        return "", str(segment or "").strip()
    return match.group("number"), match.group("body").strip(" :-")


def _extract_section_heading_segments(
    question: str,
    docs: List[Dict[str, Any]],
    limit: int = 3,
) -> List[Dict[str, str]]:
    """When the question names a section heading, return the content below it."""
    terms = _query_terms(question)
    if len(terms) < 2:
        return []

    normalized_question = _normalize_for_match(question)
    best_score = -1.0
    best_segments: List[Dict[str, str]] = []

    for doc_rank, doc in enumerate(docs[:5]):
        if not _is_kb_document(doc):
            continue

        title = _doc_title(doc)
        doc_id = _doc_identifier(doc)
        source_text = _full_kb_source_text(doc)
        split_segments = _split_kb_segments(source_text)
        if not split_segments:
            continue

        for index, segment in enumerate(split_segments):
            section_number, body = _section_number_and_body(segment)
            if not section_number or not body:
                continue

            clean_heading = _clean_kb_answer_segment(segment)
            normalized_heading = _normalize_for_match(clean_heading)
            heading_terms = _query_terms(clean_heading)
            if not heading_terms:
                continue
            if len(clean_heading) > 120:
                continue

            overlap = terms & heading_terms
            if len(overlap) < min(2, len(heading_terms)):
                continue
            if not any(term in normalized_heading for term in ("gecis", "ortam", "canli", "disaster", "failover")):
                continue

            selected: List[Dict[str, str]] = []
            child_prefix = f"{section_number}."
            for child_segment in split_segments[index + 1:]:
                child_number, _ = _section_number_and_body(child_segment)
                if child_number and not child_number.startswith(child_prefix):
                    break
                if child_number and child_number == section_number:
                    continue

                cleaned_child = _clean_kb_answer_segment(child_segment)
                if not cleaned_child:
                    continue
                if child_number and child_number.startswith(child_prefix):
                    selected.append(
                        {
                            "segment": f"{clean_heading}: {cleaned_child}",
                            "title": title,
                            "doc_id": doc_id,
                        }
                    )
                if len(selected) >= limit:
                    break

            if selected:
                score = float(len(overlap))
                if normalized_heading and normalized_heading in normalized_question:
                    score += 10.0
                if any(term in overlap for term in ("disasterdan", "canlidan")):
                    score += 1.0
                score -= doc_rank * 0.1
                if score > best_score:
                    best_score = score
                    best_segments = selected

    return best_segments


def _extract_relevant_kb_segments(question: str, docs: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, str]]:
    terms = _query_terms(question)
    terminal_question = any(term.startswith("terminal") for term in terms)
    adisyon_question = "adisyon" in terms
    mobile_printer_question = "mobil" in terms and any(term.startswith("yazici") for term in terms)

    design_principle_segments = _extract_design_principle_segments(question, docs)
    if design_principle_segments:
        return design_principle_segments

    if _is_desktop_laptop_maintenance_question(question):
        maintenance_segments = _extract_desktop_laptop_maintenance_segments(docs)
        if maintenance_segments:
            return maintenance_segments

    if _is_director_responsibility_question(question):
        director_segments = _extract_director_responsibility_segments(docs)
        if director_segments:
            return director_segments

    if _is_erp_question(question):
        erp_segments = _extract_erp_segments(docs)
        if erp_segments:
            return erp_segments

    if _is_action_plan_question(question):
        for doc in docs[:5]:
            if not _is_kb_document(doc):
                continue
            normalized_doc = _normalize_for_match(_doc_title(doc) + " " + _doc_solution_text(doc)[:500])
            if "aksiyon" in normalized_doc and "plan" in normalized_doc:
                action_plan_segments = _extract_action_plan_segments(doc)
                if action_plan_segments:
                    return action_plan_segments

    section_heading_segments = _extract_section_heading_segments(question, docs, limit=limit)
    if section_heading_segments:
        return section_heading_segments

    candidates: List[Dict[str, Any]] = []

    for doc_rank, doc in enumerate(docs[:5]):
        if not _is_kb_document(doc):
            continue
        doc_candidates: List[Dict[str, Any]] = []
        title = _doc_title(doc)
        doc_id = _doc_identifier(doc)
        source_text = _doc_solution_text(doc)
        for segment in _split_kb_segments(source_text):
            normalized_segment = _normalize_for_match(segment)
            if terminal_question and "terminal" not in normalized_segment:
                continue
            if terminal_question and not adisyon_question and "adisyon" in normalized_segment:
                continue
            if mobile_printer_question and not ("mobil" in normalized_segment and "yazici" in normalized_segment):
                continue
            overlap = sum(1 for term in terms if term in normalized_segment)
            if overlap <= 0:
                continue
            score = overlap + max(0, 5 - doc_rank) * 0.2
            if any(term in normalized_segment for term in ("terminal", "point mobile", "xplore")):
                score += 1.5
            if mobile_printer_question and "lukhan sewoo" in normalized_segment:
                score += 2.0
            doc_candidates.append(
                {
                    "score": score,
                    "segment": segment,
                    "title": title,
                    "doc_id": doc_id,
                }
            )
        if doc_rank == 0 and doc_candidates:
            candidates = doc_candidates
            break
        candidates.extend(doc_candidates)

    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected: List[Dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned_segment = _clean_kb_answer_segment(candidate["segment"])
        if not cleaned_segment:
            continue
        key = _normalize_for_match(cleaned_segment)[:180]
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "segment": cleaned_segment,
                "title": candidate["title"],
                "doc_id": candidate["doc_id"],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _build_direct_kb_answer_tr(question: str, docs: List[Dict[str, Any]]) -> str:
    segments = _extract_relevant_kb_segments(question, docs)

    if not segments:
        return (
            f"Sorunuz: {question}\n\n"
            "Kaynaklarda bu soruya doğrudan karşılık gelen bir bilgi bulunamadı."
        )

    if _is_product_name_question(question) and not any(_has_specific_named_value(item["segment"]) for item in segments):
        answer_parts = [
            f"Sorunuz: {question}\n",
            "\n**Kısa cevap:**\n",
            "Kaynaklarda bu soru için açık bir ürün/model adı bulunamadı.\n",
            "\n**Kaynakta geçen en yakın ifadeler:**\n",
        ]
        for item in segments:
            answer_parts.append(f"- {item['segment']}\n")
    else:
        answer_parts = [
            f"Sorunuz: {question}\n",
            "\n**Kısa cevap:**\n",
            "Kaynakta geçen bilgiye göre aşağıdaki noktalar öne çıkıyor.\n",
            "\n**Kaynakta geçenler:**\n",
        ]
        for item in segments:
            answer_parts.append(f"- {item['segment']}\n")

    source_refs: List[str] = []
    seen_sources: set[str] = set()
    for item in segments:
        source_key = f"{item['title']} ({item['doc_id']})"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            source_refs.append(source_key)

    if len(source_refs) == 1:
        answer_parts.append(f"\n**Kullanılan kaynak:** {source_refs[0]}")
    else:
        answer_parts.append("\n**Kullanılan kaynaklar:**\n")
        for source_ref in source_refs:
            answer_parts.append(f"- {source_ref}\n")
    return "".join(answer_parts)


def _build_direct_kb_answer_en(question: str, docs: List[Dict[str, Any]]) -> str:
    segments = _extract_relevant_kb_segments(question, docs)

    if not segments:
        return (
            f"Your question: {question}\n\n"
            "No directly matching information was found in the available sources."
        )

    if _is_product_name_question(question) and not any(_has_specific_named_value(item["segment"]) for item in segments):
        answer_parts = [
            f"Your question: {question}\n",
            "\nNo explicit product/model name was found for this question in the sources. Closest source statements:\n",
        ]
        for item in segments:
            answer_parts.append(f"- {item['segment']}\n")
    else:
        answer_parts = [
            f"Your question: {question}\n",
            "\nAccording to the retrieved source:\n",
        ]
        for item in segments:
            answer_parts.append(f"- {item['segment']}\n")

    source_refs: List[str] = []
    seen_sources: set[str] = set()
    for item in segments:
        source_key = f"{item['title']} ({item['doc_id']})"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            source_refs.append(source_key)

    if len(source_refs) == 1:
        answer_parts.append(f"\nSource: {source_refs[0]}")
    else:
        answer_parts.append("\nSources:\n")
        for source_ref in source_refs:
            answer_parts.append(f"- {source_ref}\n")
    return "".join(answer_parts)


def _docs_for_direct_segments(docs: List[Dict[str, Any]], segments: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Keep source metadata aligned with the source snippets used in direct answers."""
    segment_doc_ids = {str(item.get("doc_id", "") or "").strip() for item in segments if item.get("doc_id")}
    segment_titles = {
        _normalize_for_match(str(item.get("title", "") or ""))
        for item in segments
        if item.get("title")
    }
    if not segment_doc_ids and not segment_titles:
        return docs

    def id_family(identifier: str) -> set[str]:
        values = {identifier} if identifier else set()
        if "_chunk_" in identifier:
            values.add(identifier.split("_chunk_", 1)[0])
        return values

    segment_id_families: set[str] = set()
    for identifier in segment_doc_ids:
        segment_id_families.update(id_family(identifier))

    filtered: List[Dict[str, Any]] = []
    for doc in docs:
        doc_ids = id_family(_doc_identifier(doc))
        doc_ids.update(id_family(_doc_document_id(doc)))
        title = _normalize_for_match(_doc_title(doc))
        id_matches = bool(doc_ids & segment_id_families)
        title_matches = title in segment_titles if title else False
        if id_matches or title_matches:
            matched_segment_texts: List[str] = []
            for item in segments:
                item_ids = id_family(str(item.get("doc_id", "") or "").strip())
                item_title = _normalize_for_match(str(item.get("title", "") or ""))
                if bool(doc_ids & item_ids) or (title and item_title == title):
                    segment_text = str(item.get("segment", "") or "").strip()
                    if segment_text:
                        matched_segment_texts.append(segment_text)

            aligned_doc = doc.copy()
            if matched_segment_texts:
                aligned_text = " ".join(matched_segment_texts)
                aligned_doc["text"] = aligned_text
                aligned_doc["content"] = aligned_text
                aligned_doc["description"] = aligned_text
            filtered.append(aligned_doc)

    return filtered or docs


def _combined_doc_text(question: str, docs: List[Dict[str, Any]]) -> str:
    pieces = [question]
    for doc in docs[:3]:
        pieces.extend(
            [
                str(doc.get("category", "") or ""),
                str(doc.get("subcategory", "") or ""),
                _doc_title(doc),
                _doc_solution_text(doc),
            ]
        )
    return " ".join(piece.casefold() for piece in pieces if piece)


def _infer_support_scenario(question: str, docs: List[Dict[str, Any]]) -> str:
    text = _combined_doc_text(question, docs)
    normalized_text = _normalize_for_match(text)

    if any(term in text for term in ("vpn", "forticlient", "remote access", "uzaktan erişim")):
        return "vpn"
    if any(term in normalized_text for term in ("disk alani", "disk dol", "depolama", "bos alan", "storage")):
        return "storage"
    if any(term in normalized_text for term in ("laptop", "dizustu", "masaustu", "yavas", "performans", "bellek", "ram")):
        return "performance"
    if any(term in text for term in ("mail", "e-posta", "outlook", "exchange", "posta kutusu", "kota")):
        return "mail"
    if any(term in text for term in ("mfa", "authenticator", "token", "şifre", "parola", "sso", "hesap kilit")):
        return "identity"
    if any(term in text for term in ("yazıcı", "printer", "çıktı", "toner", "tarama")):
        return "printer"
    if any(term in text for term in ("teams", "mikrofon", "kamera", "toplantı", "ses")):
        return "teams"
    return "generic"


def _support_scenario_from_text(text: str) -> str:
    normalized_text = _normalize_for_scenario_match(text)

    if any(term in normalized_text for term in ("vpn", "forticlient", "remote access", "uzaktan erisim")):
        return "vpn"
    if any(term in normalized_text for term in ("disk alani", "disk dol", "depolama", "bos alan", "storage")):
        return "storage"
    if any(term in normalized_text for term in ("laptop", "dizustu", "masaustu", "yavas", "performans", "bellek", "ram")):
        return "performance"
    if any(term in normalized_text for term in ("mail", "email", "e posta", "outlook", "exchange", "posta kutusu", "kota")):
        return "mail"
    if any(term in normalized_text for term in ("mfa", "authenticator", "token", "sifre", "parola", "sso", "hesap kilit", "login", "giris")):
        return "identity"
    if any(term in normalized_text for term in ("yazici", "printer", "cikti", "toner", "tarama", "spooler")):
        return "printer"
    if any(term in normalized_text for term in ("teams", "mikrofon", "kamera", "toplanti", "ses")):
        return "teams"
    return "generic"


def _infer_support_scenario(question: str, docs: List[Dict[str, Any]]) -> str:
    question_scenario = _support_scenario_from_text(question)
    if question_scenario != "generic":
        return question_scenario
    return _support_scenario_from_text(_combined_doc_text("", docs))


def _doc_matches_support_scenario(doc: Dict[str, Any], scenario: str) -> bool:
    if scenario == "generic":
        return True
    return _support_scenario_from_text(_combined_doc_text("", [doc])) == scenario


def _filter_advisory_docs_for_question(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep advisory answers tied to sources that match the user's support scenario."""
    scenario = _infer_support_scenario(question, [])
    if scenario == "generic":
        return docs
    return [doc for doc in docs if _doc_matches_support_scenario(doc, scenario)]


def _build_support_playbook_doc(question: str, scenario: str, language: str) -> Dict[str, Any]:
    """Create a clearly labeled fallback source when retrieval has no scenario match."""
    if scenario == "generic":
        return {}

    if language == "tr":
        title_by_scenario = {
            "vpn": "VPN genel BT kontrol listesi",
            "storage": "Disk alanı genel BT kontrol listesi",
            "performance": "Performans genel BT kontrol listesi",
            "mail": "E-posta genel BT kontrol listesi",
            "identity": "Kimlik ve parola genel BT kontrol listesi",
            "printer": "Yazıcı genel BT kontrol listesi",
            "teams": "Teams genel BT kontrol listesi",
        }
        title = title_by_scenario.get(scenario, "Genel BT kontrol listesi")
        steps = _action_steps_tr(question, [])
        note = (
            "İndekste bu senaryoya özel güvenilir kayıt bulunamadı. "
            "Aşağıdaki maddeler genel BT destek kontrol listesinden üretilmiştir."
        )
    else:
        title_by_scenario = {
            "vpn": "VPN general IT checklist",
            "storage": "Storage general IT checklist",
            "performance": "Performance general IT checklist",
            "mail": "Email general IT checklist",
            "identity": "Identity and password general IT checklist",
            "printer": "Printer general IT checklist",
            "teams": "Teams general IT checklist",
        }
        title = title_by_scenario.get(scenario, "General IT checklist")
        steps = _action_steps_en(question, [])
        note = (
            "No reliable indexed source matched this support scenario. "
            "The items below come from the general IT support checklist."
        )

    content = f"{note}\n" + "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return {
        "doc_id": f"playbook_{scenario}",
        "id": f"playbook_{scenario}",
        "doc_type": "playbook",
        "title": title,
        "short_description": title,
        "resolution": content,
        "text": content,
        "description": content,
        "score": 0.55,
    }


def _action_steps_tr(question: str, docs: List[Dict[str, Any]]) -> List[str]:
    scenario = _infer_support_scenario(question, docs)

    if scenario == "vpn":
        return [
            "İnternet bağlantısını ve VPN dışında web erişiminin çalışıp çalışmadığını kontrol edin.",
            "FortiClient/VPN istemcisinde kullanıcı adı, parola, MFA bildirimi ve hata kodunu kontrol edin.",
            "Şirket içi kaynaklara erişim yoksa VPN profilini/gateway bilgisini yenileyip yeniden bağlanmayı deneyin.",
            "Sorun sürerse saat, hata kodu, ekran görüntüsü ve etkilenen kullanıcı bilgisini BT ekibine iletin.",
        ]
    if scenario == "mail":
        return [
            "Outlook Web veya farklı bir cihazdan mail gönderimini deneyerek sorunun istemci mi servis mi olduğunu ayırın.",
            "Posta kutusu kotasını ve ek boyutu limitlerini kontrol edin; kota doluysa eski/iri iletileri arşivleyin.",
            "Exchange bağlantısı, mail flow ve varsa NDR/hata kodu bilgisini not edin.",
            "Sorun devam ederse BT ekibinden kullanıcı mailbox, connector ve servis sağlığı kontrollerini istemek gerekir.",
        ]
    if scenario == "storage":
        return [
            "Disk alanını kontrol edin; gereksiz geçici dosya, indirilenler ve eski kurulum dosyaları için temizlik yapın.",
            "Büyük dosya ve klasörleri belirleyip kurum politikasına uygunsa arşivleyin veya güvenli silme işlemi uygulayın.",
            "Geri Dönüşüm Kutusu, tarayıcı önbelleği ve uygulama cache alanlarını temizleyerek boş alan kazanmaya çalışın.",
            "Disk alanı tekrar hızla doluyorsa dosya yolu, hata mesajı ve kullanıcı bilgisini BT ekibine iletin.",
        ]
    if scenario == "performance":
        return [
            "Laptop performans sorunu için CPU, disk kullanımı ve bellek doluluk oranını Görev Yöneticisi'nden kontrol edin.",
            "Disk temizlik işlemi yapın, gereksiz başlangıç uygulamalarını kapatın ve yeterli boş alan olduğunu doğrulayın.",
            "Windows, sürücü ve güvenlik güncelleme durumunu kontrol ederek eksik güncellemeleri planlı şekilde tamamlayın.",
            "Sorun sürerse cihaz modeli, bellek miktarı, disk sağlığı ve yavaşlığın ne zaman başladığını BT ekibine iletin.",
        ]
    if scenario == "identity":
        return [
            "Parola süresi, hesap kilidi ve MFA cihaz kaydı durumunu kontrol edin.",
            "Self servis parola sıfırlama veya kurum kimlik portalı üzerinden oturumu yenilemeyi deneyin.",
            "Authenticator saat senkronu, push bildirimi ve alternatif doğrulama yöntemlerini kontrol edin.",
            "Şüpheli MFA isteği veya beklenmeyen oturum varsa parolayı değiştirip BT/SOC ekibine bildirin.",
        ]
    if scenario == "printer":
        return [
            "Yazıcının açık, ağda erişilebilir ve doğru kuyruk üzerinden seçili olduğunu kontrol edin.",
            "Yazdırma kuyruğunu temizleyip test sayfası göndermeyi deneyin.",
            "Sürücü, toner/kağıt durumu ve kullanıcı yetkisini kontrol edin.",
            "Sorun sürerse yazıcı adı, hata mesajı ve etkilenen kullanıcıları BT ekibine iletin.",
        ]
    if scenario == "teams":
        return [
            "Teams web/masaüstü ayrımı yaparak sorunun uygulama mı cihaz mı olduğunu kontrol edin.",
            "Mikrofon, kamera ve hoparlör aygıt seçimlerini doğrulayın.",
            "Oturumu kapatıp açın, Teams önbelleğini temizleyin ve test toplantısı yapın.",
            "Sorun devam ederse toplantı zamanı, cihaz modeli ve hata ekranını BT ekibine iletin.",
        ]

    return [
        "Hata mesajı, etkilenen uygulama, cihaz ve sorunun başladığı zamanı not edin.",
        "Aynı işlemi farklı tarayıcı/cihaz/ağ üzerinden deneyerek kapsamı daraltın.",
        "Benzer geçmiş kayıtların çözüm adımlarını güvenli olanlardan başlayarak uygulayın.",
        "İş kesintisi devam ederse bulgularla birlikte BT destek ekibine eskale edin.",
    ]


def _action_steps_en(question: str, docs: List[Dict[str, Any]]) -> List[str]:
    scenario = _infer_support_scenario(question, docs)

    if scenario == "vpn":
        return [
            "Check whether general internet access works outside the VPN.",
            "Verify VPN username, password, MFA prompt, client profile, and any error code.",
            "Reconnect after refreshing the VPN profile or gateway settings if internal resources are unreachable.",
            "If it continues, send the timestamp, error code, screenshot, and affected user details to IT.",
        ]
    if scenario == "mail":
        return [
            "Try sending from Outlook Web or another device to separate client-side and service-side issues.",
            "Check mailbox quota and attachment size limits; archive or remove large messages if needed.",
            "Record Exchange connectivity, mail-flow, and NDR/error-code details.",
            "If it continues, ask IT to check the mailbox, connector, and service health.",
        ]
    if scenario == "storage":
        return [
            "Check disk space and clean temporary files, downloads, and old installer files.",
            "Identify large files and folders, then archive them or delete them according to company policy.",
            "Empty the recycle bin, browser cache, and application cache to recover storage space.",
            "If disk space fills up again quickly, send the file path, error message, and user details to IT.",
        ]
    if scenario == "performance":
        return [
            "Check laptop performance in Task Manager, especially CPU, disk usage, and memory consumption.",
            "Run disk cleanup, disable unnecessary startup apps, and verify that enough free space remains.",
            "Check Windows, driver, and security update status and complete missing updates in a planned way.",
            "If it continues, send the device model, memory amount, disk health, and when the slowdown started to IT.",
        ]
    if scenario == "identity":
        return [
            "Check password expiry, account lockout, and MFA device registration.",
            "Retry through the self-service password or identity portal.",
            "Verify authenticator time sync, push notifications, and alternate verification methods.",
            "For suspicious MFA prompts, change the password and notify IT/SOC.",
        ]
    return [
        "Collect the exact error message, application/device, and start time.",
        "Try the same action from another browser, device, or network to narrow the scope.",
        "Apply the safest matching steps from similar historical tickets first.",
        "Escalate to IT support with the collected evidence if the interruption continues.",
    ]


def _message_content(message: Dict[str, Any]) -> str:
    return str(message.get("content") or message.get("text") or "").strip()


def _previous_user_issue(conversation_history: Optional[List[Dict[str, Any]]]) -> str:
    for message in reversed(conversation_history or []):
        if message.get("role") != "user":
            continue
        content = _message_content(message)
        if content:
            return content
    return ""


def _is_follow_up_question(question: str) -> bool:
    if _is_acknowledgment_message(question):
        return False

    normalized = question.casefold()
    ascii_normalized = _normalize_for_match(question)
    words = normalized.split()
    explicit_kb_markers = (
        "ozdilek",
        "talimat",
        "dokuman",
        "belge",
        "prosedur",
        "yonerge",
        "platform",
        "yapay zeka",
    )
    explicit_document_lookup_markers = (
        "belge",
        "dokuman",
        "elektronik",
        "elekronik",
        "gomulu",
        "ilke",
        "ilkeleri",
        "platform",
        "prosedur",
        "talimat",
        "tasarim",
        "yapay zeka",
        "yazilim",
        "yonerge",
    )
    clear_correction_markers = (
        "az once",
        "ben sadece",
        "bunu sordum",
        "dedigin",
        "demek istedim",
        "kastettim",
        "onceki",
        "sadece",
        "sordum",
        "takip",
        "yukaridaki",
    )
    direct_follow_up_markers = (
        "olmadi",
        "olmadı",
        "devam",
        "bunlar",
        "bunu",
        "nereden",
        "nerde",
        "adim",
        "adım",
        "soyledigin",
        "söylediğin",
        "uyguladim",
        "uyguladım",
        "sonra ne",
    )
    normalized_follow_up_markers = (
        "az once",
        "ben sadece",
        "bunu sordum",
        "dedigin",
        "demek istedim",
        "kastettim",
        "onceki",
        "sadece",
        "sordum",
        "soyledigin",
        "takip",
        "yukaridaki",
    )
    standalone_it_terms = (
        "vpn",
        "mail",
        "exchange",
        "outlook",
        "teams",
        "mfa",
        "sifre",
        "şifre",
        "parola",
        "yazici",
        "yazıcı",
        "printer",
        "internet",
        "wifi",
    )

    has_explicit_kb_target = len(words) > 5 and any(marker in ascii_normalized for marker in explicit_kb_markers)
    has_explicit_document_lookup = len(words) > 5 and any(marker in ascii_normalized for marker in explicit_document_lookup_markers)
    has_clear_correction = any(marker in ascii_normalized for marker in clear_correction_markers)
    if has_explicit_document_lookup:
        return False
    if has_explicit_kb_target and not has_clear_correction:
        return False

    if any(marker in normalized for marker in direct_follow_up_markers):
        return True

    if any(marker in ascii_normalized for marker in normalized_follow_up_markers):
        return True

    if "nasil" in ascii_normalized and len(words) <= 5:
        return True

    return len(words) <= 5 and not any(term in normalized or term in ascii_normalized for term in standalone_it_terms)


def _build_contextual_question(
    question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Resolve short corrections/follow-ups against the latest user issue."""
    if not conversation_history or not _is_follow_up_question(question):
        return question

    previous_issue = _previous_user_issue(conversation_history)
    if not previous_issue:
        return question

    if len(previous_issue) > 240:
        previous_issue = previous_issue[:240]

    return f"{previous_issue}\nTakip sorusu/duzeltme: {question}"

def _build_contextual_retrieval_query(
    question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Use the previous user issue for short follow-up questions without changing the visible answer."""
    return _build_contextual_question(question, conversation_history)


def _build_advisory_answer_tr(question: str, docs: List[Dict[str, Any]]) -> str:
    """
    Build Turkish advisory-style answer from past ticket examples.
    NOW WITH DETAILED STEP-BY-STEP FORMATTING (PHASE 7.5).
    
    Args:
        question: User's question
        docs: Retrieved documents (past tickets and PDF pages)
        
    Returns:
        Advisory answer in Turkish with detailed step-by-step instructions
    """
    is_playbook_fallback = bool(docs) and all(
        str(doc.get("doc_type", "") or "").casefold() == "playbook"
        for doc in docs
    )

    answer_parts = [f"Sorunuz: {question}\n"]
    if is_playbook_fallback:
        answer_parts.append(
            "\n**Kaynak eşleşmesi:**\n"
            "- İndekste bu senaryoya özel güvenilir kayıt bulunamadı; aşağıdaki yanıt genel BT kontrol listesidir.\n"
        )

    answer_parts.append("\n**Sizin deneyebileceğiniz adımlar:**\n")

    for index, step in enumerate(_action_steps_tr(question, docs), 1):
        answer_parts.append(f"{index}. {step}\n")

    if is_playbook_fallback:
        answer_parts.append("\n**Yanıtın dayanağı:**\n")
    else:
        answer_parts.append("\n**Geçmiş benzer kayıtlarda incelenenler:**\n")
    
    examples_rendered = 0
    if is_playbook_fallback:
        answer_parts.append("- Genel BT kontrol listesi: playbook\n")
        examples_rendered = len(docs)
    else:
        # Show top 3 usable examples. Ticket documents use resolution; KB chunks use text/content.
        for doc in docs:
            if examples_rendered >= 3:
                break

            ticket_id = _doc_identifier(doc)
            doc_type = doc.get("doc_type", "itsm_ticket")
            short_desc = _doc_title(doc)
            resolution = _doc_solution_text(doc)

            if short_desc and resolution:
                examples_rendered += 1
                source_label = _source_label_tr(doc_type)
                answer_parts.append(f"\n**Örnek {examples_rendered} - {source_label}: {ticket_id}**\n")
                answer_parts.append(f"- **Durum:** {short_desc}\n")

                formatted_resolution = _format_resolution_text(resolution, doc_type)
                answer_parts.append(f"- **Uygulanan Çözüm:**\n{formatted_resolution}\n")
    
    answer_parts.append("\n**Ne zaman BT ekibine iletilmeli?**\n")
    answer_parts.append("- Yukarıdaki kontrollerden sonra sorun sürüyorsa hata kodu, saat, ekran görüntüsü ve etkilenen kullanıcı bilgisini ekleyin.\n")
    answer_parts.append("- Benzer geçmiş adımları referans göstererek BT destek ekibinden kontrol veya uygulama talep edebilirsiniz.\n")
    
    if not is_playbook_fallback and len(docs) > examples_rendered:
        answer_parts.append(
            f"\n\n(Toplam {len(docs)} benzer durum bulundu)"
        )
    
    logger.debug("advisory_answer_generated_tr", 
                question=question[:50],
                num_docs=len(docs),
                num_examples=examples_rendered)
    
    return "".join(answer_parts)


def _format_resolution_text(text: str, doc_type: str) -> str:
    """
    Format resolution text to highlight step-by-step instructions.
    
    Args:
        text: Raw resolution text
        doc_type: Type of document (document for PDF, itsm_ticket for ticket)
        
    Returns:
        Formatted text with clear step-by-step structure
    """
    if not text or len(text) < 20:
        return text
    
    # For PDF documents, show more content (up to 1500 chars for detailed instructions)
    if doc_type == "document":
        max_length = 1500
    else:
        max_length = 800
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Split into lines
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect and format numbered steps
        if any(line.lower().startswith(f"{num}.") or line.lower().startswith(f"{num})") 
               for num in range(1, 20)):
            formatted_lines.append(line)
        
        # Detect and format bullet points
        elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
            formatted_lines.append(line)
        
        # Detect keywords for steps
        elif any(keyword in line.lower() for keyword in ['adım', 'işlem', 'kontrol edin', 'yapınız', 'tıklayın']):
            formatted_lines.append(f"- {line}")
        
        # Regular lines
        else:
            formatted_lines.append(f"- {line}")
    
    return '\n'.join(formatted_lines)


def _build_advisory_answer_en(question: str, docs: List[Dict[str, Any]]) -> str:
    """
    Build English advisory-style answer from past ticket examples.
    NOW WITH DETAILED STEP-BY-STEP FORMATTING (PHASE 7.5).
    
    Args:
        question: User's question
        docs: Retrieved documents (past tickets and PDF pages)
        
    Returns:
        Advisory answer in English with detailed step-by-step instructions
    """
    is_playbook_fallback = bool(docs) and all(
        str(doc.get("doc_type", "") or "").casefold() == "playbook"
        for doc in docs
    )

    answer_parts = [f"Your question: {question}\n"]
    if is_playbook_fallback:
        answer_parts.append(
            "\n**Source match:**\n"
            "- No reliable indexed record matched this scenario; the answer below uses the general IT checklist.\n"
        )

    answer_parts.append("\n**Steps you can try first:**\n")

    for index, step in enumerate(_action_steps_en(question, docs), 1):
        answer_parts.append(f"{index}. {step}\n")

    if is_playbook_fallback:
        answer_parts.append("\n**Answer basis:**\n")
    else:
        answer_parts.append("\n**What was checked in similar past tickets:**\n")
    
    examples_rendered = 0
    if is_playbook_fallback:
        answer_parts.append("- General IT checklist: playbook\n")
        examples_rendered = len(docs)
    else:
        # Show top 3 usable examples. Ticket documents use resolution; KB chunks use text/content.
        for doc in docs:
            if examples_rendered >= 3:
                break

            ticket_id = _doc_identifier(doc)
            doc_type = doc.get("doc_type", "itsm_ticket")
            short_desc = _doc_title(doc)
            resolution = _doc_solution_text(doc)

            if short_desc and resolution:
                examples_rendered += 1
                source_label = _source_label_en(doc_type)
                answer_parts.append(f"\n**Example {examples_rendered} - {source_label}: {ticket_id}**\n")
                answer_parts.append(f"- **Issue:** {short_desc}\n")

                formatted_resolution = _format_resolution_text(resolution, doc_type)
                answer_parts.append(f"- **Resolution Applied:**\n{formatted_resolution}\n")
    
    answer_parts.append("\n**When to escalate to IT:**\n")
    answer_parts.append("- If the issue continues after these checks, include the error code, time, screenshot, and affected user details.\n")
    answer_parts.append("- You can reference similar past steps when asking IT support to verify or apply the fix.\n")
    
    if not is_playbook_fallback and len(docs) > examples_rendered:
        answer_parts.append(
            f"\n\n({len(docs)} similar cases found in total)"
        )
    
    logger.debug("advisory_answer_generated_en", 
                question=question[:50],
                num_docs=len(docs),
                num_examples=examples_rendered)
    
    return "".join(answer_parts)


class RAGPipeline:
    """
    Main RAG pipeline that coordinates retrieval and generation
    with strict "no source, no answer" policy.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        prompt_builder: Optional[PromptBuilder] = None,
        confidence_estimator: Optional[ConfidenceEstimator] = None,
        llm_model=None,  # Will be transformers model or API client
        max_context_length: int = 2048,
        confidence_threshold: float = 0.7,
        # PHASE 8: Real LLM settings
        use_real_llm: bool = False,
        openai_api_key: Optional[str] = None,
        llm_model_name: str = "gpt-4o-mini",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 1500
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Hybrid retriever for document retrieval
            prompt_builder: Prompt builder (creates default if None)
            confidence_estimator: Confidence estimator (creates default if None)
            llm_model: LLM model for generation
            max_context_length: Maximum context length for prompts
            confidence_threshold: Minimum confidence for answers
            use_real_llm: Whether to use real LLM (True) or stub (False)
            openai_api_key: OpenAI API key for real LLM
            llm_model_name: OpenAI model name (gpt-4o-mini, gpt-4o, etc.)
            llm_temperature: LLM temperature for generation
            llm_max_tokens: Maximum tokens for LLM response
        """
        self.retriever = retriever
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator(confidence_threshold)
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        self.confidence_threshold = confidence_threshold
        
        # PHASE 8: Real LLM settings
        self.use_real_llm = use_real_llm
        self.openai_api_key = openai_api_key
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens

        if self.use_real_llm and not OPENAI_AVAILABLE:
            logger.warning(
                "real_llm_disabled_openai_package_missing",
                message="USE_REAL_LLM is true, but the openai package is not installed. Falling back to stub answers.",
            )
            self.use_real_llm = False
        elif self.use_real_llm and not self.openai_api_key:
            logger.warning(
                "real_llm_disabled_api_key_missing",
                message="USE_REAL_LLM is true, but OPENAI_API_KEY is missing. Falling back to stub answers.",
            )
            self.use_real_llm = False
        
        # IT relevance checker for filtering non-IT queries
        self.it_relevance_checker = ITRelevanceChecker()
        
        logger.info("rag_pipeline_initialized",
                   max_context_length=max_context_length,
                   confidence_threshold=confidence_threshold,
                   use_real_llm=self.use_real_llm,
                   llm_model=llm_model_name if self.use_real_llm else "stub")
    
    def answer(
        self,
        question: str,
        *,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> RAGResult:
        """
        Answer a question using the RAG pipeline (PHASE 4 implementation).
        
        This is the main entry point for the RAG system that:
        1. Retrieves relevant documents using HybridRetriever
        2. Computes retrieval confidence
        3. Applies "no source, no answer" policy
        4. Generates answer using LLM stub (or real LLM in production)
        5. Returns structured RAGResult
        
        PHASE 9: Now supports conversation history for context-aware answers!
        
        Args:
            question: User's question
            language: Language code (e.g., "tr", "en"), auto-detected if None
            session_id: Optional session ID for conversation tracking
            conversation_history: Previous messages for context (PHASE 9)
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResult with answer, confidence, sources, and metadata
            
        Example:
            >>> pipeline = RAGPipeline(retriever, ...)
            >>> result = pipeline.answer("Outlook şifremi unuttum")
            >>> print(result.answer)
            >>> print(f"Confidence: {result.confidence}")
        """
        logger.info("rag_answer_request", 
                   question=question[:100],
                   language=language,
                   session_id=session_id)
        
        # Auto-detect language if not provided or if an unsupported value is supplied.
        if language not in {"tr", "en"}:
            language = self._detect_language(question)
        
        # Step 0.5: Handle thank you messages and acknowledgments
        if _is_acknowledgment_message(question):
            # Thank you / positive feedback messages - return friendly acknowledgment
            if language == "tr":
                answer = "Rica ederim! Başka bir konuda yardımcı olabilir miyim?"
            else:
                answer = "You're welcome! Is there anything else I can help you with?"

            return RAGResult(
                answer=answer,
                confidence=0.0,
                sources=[],
                has_answer=False,
                language=language,
                intent="acknowledgment",
                retrieved_docs=[],
                debug_info={"rejection_reason": "acknowledgment_message"}
            )
        
        # Step 0: Check if query is IT-related (filter non-IT queries)
        # IMPORTANT: Check conversation history - if previous messages were IT-related,
        # Check if query should be rejected (non-IT)
        is_it, it_confidence = self.it_relevance_checker.is_it_related(question)
        should_reject = self.it_relevance_checker.should_reject_query(question)
        
        # If query is explicitly non-IT (high confidence, e.g., "şişe", "yemek"), 
        # reject immediately regardless of conversation history
        if should_reject and it_confidence >= 0.8:
            # Explicit non-IT keyword detected - reject immediately
            logger.info("query_rejected_explicit_non_it", 
                       question=question[:50],
                       confidence=it_confidence)
            # Return rejection immediately - don't process further
            if language == "tr":
                answer = "Üzgünüm, bu soru BT (Bilgi Teknolojileri) destek konularıyla ilgili değil. Lütfen bilgisayar, yazılım, ağ, güvenlik veya diğer BT konularıyla ilgili sorularınızı sorun."
            else:
                answer = "I'm sorry, this question is not related to IT (Information Technology) support topics. Please ask questions about computers, software, networks, security, or other IT-related topics."
            
            return RAGResult(
                answer=answer,
                confidence=0.0,
                sources=[],
                has_answer=False,
                language=language,
                intent=None,
                retrieved_docs=[],
                debug_info={"rejection_reason": "explicit_non_it_query", "confidence": it_confidence}
            )
        # If query seems non-IT but with lower confidence, check conversation history
        elif should_reject and conversation_history:
            # Check if any previous message in conversation was IT-related
            has_it_context = False
            for msg in conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    content = msg.get("content", "")
                    is_it_hist, _ = self.it_relevance_checker.is_it_related(content)
                    if is_it_hist:
                        has_it_context = True
                        logger.debug("it_context_found_in_history", 
                                   message_preview=content[:50])
                        break
            
            # If conversation has IT context, don't reject follow-up questions
            if has_it_context:
                should_reject = False
                logger.info("query_accepted_due_to_it_context", 
                           question=question[:50],
                           has_history=True)
        
        if should_reject:
            logger.info("query_rejected_non_it", question=question[:100])
            if language == "tr":
                answer = "Üzgünüm, bu soru BT (Bilgi Teknolojileri) destek konularıyla ilgili değil. Lütfen bilgisayar, yazılım, ağ, güvenlik veya diğer BT konularıyla ilgili sorularınızı sorun."
            else:
                answer = "I'm sorry, this question is not related to IT (Information Technology) support topics. Please ask questions about computers, software, networks, security, or other IT-related topics."
            
            return RAGResult(
                answer=answer,
                confidence=0.0,
                sources=[],
                has_answer=False,
                language=language,
                intent=None,
                retrieved_docs=[],
                debug_info={"rejection_reason": "non_it_query"}
            )
        
        # Step 1: Retrieve relevant documents
        contextual_question = _build_contextual_question(question, conversation_history)
        retrieval_question = _build_contextual_retrieval_query(question, conversation_history)
        retrieved_docs = self.retriever.search(retrieval_question, top_k=top_k)
        
        # Collect debug info from retrieval
        debug_info = {}
        debug_info["used_conversation_context"] = retrieval_question != question
        if retrieved_docs:
            # Get alpha_used from first result (all should have same alpha)
            first_doc = retrieved_docs[0]
            debug_info["alpha_used"] = first_doc.get("alpha_used")
            
            # Get actual source counts from metadata (added by hybrid retriever)
            debug_info["bm25_results_count"] = first_doc.get("_bm25_source_count", 0)
            debug_info["embedding_results_count"] = first_doc.get("_embedding_source_count", 0)
            debug_info["hybrid_results_count"] = len(retrieved_docs)
            
            # Determine query type based on alpha
            alpha = debug_info.get("alpha_used", 0.5)
            if alpha < 0.4:
                debug_info["query_type"] = "short_technical"  # Embedding favored
            elif alpha < 0.6:
                debug_info["query_type"] = "medium"  # Balanced
            else:
                debug_info["query_type"] = "long_detailed"  # BM25 favored
        
        logger.debug("retrieval_completed", 
                    num_docs=len(retrieved_docs),
                    question=question[:50],
                    retrieval_question=retrieval_question[:120],
                    debug_info=debug_info)
        
        # Step 2: Check if we have any documents
        if not retrieved_docs:
            logger.warning("no_documents_retrieved", question=question[:100])
            return self._build_no_answer_result(
                language=language,
                reason="no_documents"
            )
        
        # Step 3: Compute retrieval confidence
        retrieval_scores = [doc.get("score", 0.0) for doc in retrieved_docs]
        top_score = max(retrieval_scores) if retrieval_scores else 0.0
        
        # If top score is too low, don't attempt to answer
        if top_score < 0.1:  # Very low threshold for retrieval
            logger.warning("low_retrieval_scores",
                         top_score=top_score,
                         question=question[:100])
            return self._build_no_answer_result(
                language=language,
                reason="low_scores",
                retrieved_docs=retrieved_docs
            )
        
        # Step 4: Generate answer using direct KB evidence, real LLM, or stub (PHASE 8)
        answer_docs = retrieved_docs
        used_direct_kb_answer = False
        used_playbook_fallback = False
        try:
            source_grounded_kb_required = _requires_source_grounded_kb_answer(contextual_question)
            direct_kb_docs = _direct_kb_docs_for_question(contextual_question, retrieved_docs)

            if source_grounded_kb_required:
                direct_segments = _extract_relevant_kb_segments(contextual_question, direct_kb_docs) if direct_kb_docs else []
                if not direct_segments:
                    logger.info("direct_kb_answer_rejected_no_evidence",
                                question=contextual_question[:100],
                                kb_doc_count=len(direct_kb_docs))
                    return self._build_no_answer_result(
                        language=language,
                        reason="no_direct_kb_evidence",
                        retrieved_docs=retrieved_docs,
                        confidence=0.0,
                        debug_info=debug_info
                    )

                answer_docs = _docs_for_direct_segments(direct_kb_docs, direct_segments)
                used_direct_kb_answer = True
                if language == "tr":
                    generated_answer = _build_direct_kb_answer_tr(contextual_question, direct_kb_docs)
                else:
                    generated_answer = _build_direct_kb_answer_en(contextual_question, direct_kb_docs)
            else:
                support_scenario = _infer_support_scenario(contextual_question, [])
                answer_docs = _filter_advisory_docs_for_question(contextual_question, retrieved_docs)
                debug_info["support_scenario"] = support_scenario
                debug_info["answer_source_count"] = len(answer_docs)

                if support_scenario != "generic" and not answer_docs:
                    logger.info("answer_rejected_no_scenario_matched_sources",
                                question=contextual_question[:100],
                                support_scenario=support_scenario)
                    playbook_doc = _build_support_playbook_doc(contextual_question, support_scenario, language)
                    if not playbook_doc:
                        return self._build_no_answer_result(
                            language=language,
                            reason="no_scenario_matched_sources",
                            retrieved_docs=retrieved_docs,
                            confidence=0.0,
                            debug_info=debug_info
                        )
                    answer_docs = [playbook_doc]
                    used_playbook_fallback = True
                    debug_info["answer_source_count"] = len(answer_docs)
                    debug_info["used_playbook_fallback"] = True

                if self.use_real_llm and self.openai_api_key:
                    generated_answer = generate_answer_with_llm(
                        question=contextual_question,
                        docs=answer_docs,
                        language=language,
                        conversation_history=conversation_history or [],  # PHASE 9
                        api_key=self.openai_api_key,
                        model=self.llm_model_name,
                        temperature=self.llm_temperature,
                        max_tokens=self.llm_max_tokens
                    )
                else:
                    generated_answer = generate_answer_with_stub(
                        question=contextual_question,
                        docs=answer_docs,
                        language=language
                    )
        except Exception as e:
            logger.error("answer_generation_failed", error=str(e))
            return self._build_no_answer_result(
                language=language,
                reason="generation_error"
            )
        
        # Step 5: Estimate confidence using the confidence estimator
        answer_retrieval_scores = [doc.get("score", 0.0) for doc in answer_docs]
        confidence, has_sufficient_confidence = self.confidence_estimator.estimate_confidence(
            answer=generated_answer,
            query=contextual_question,
            retrieved_docs=answer_docs,
            retrieval_scores=answer_retrieval_scores
        )
        
        # Step 5.5: Adjust confidence threshold for conversation history
        # If this is a follow-up question in an IT-related conversation,
        # use a lower threshold to allow more lenient answers
        effective_threshold = self.confidence_threshold
        if used_direct_kb_answer:
            # Direct KB answers copy source sentences verbatim, so a lower retrieval
            # score can still be acceptable when there is explicit evidence.
            effective_threshold = min(effective_threshold, 0.45)
        if used_playbook_fallback:
            # Playbook fallback is useful guidance, but not retrieved evidence.
            effective_threshold = min(effective_threshold, 0.45)
            confidence = min(confidence, 0.62)
        if conversation_history:
            # Check if conversation has IT context (already checked earlier)
            has_it_context = False
            for msg in conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    content = msg.get("content", "")
                    is_it, _ = self.it_relevance_checker.is_it_related(content)
                    if is_it:
                        has_it_context = True
                        break
            
            # Lower threshold for follow-up questions in IT conversations
            if has_it_context and not self.it_relevance_checker.is_it_related(question)[0]:
                # This is likely a follow-up question (e.g., "2. adımı anlamadım")
                effective_threshold = max(0.5, self.confidence_threshold - 0.15)  # Lower by 0.15, min 0.5
                logger.debug("confidence_threshold_adjusted_for_followup",
                           original_threshold=self.confidence_threshold,
                           effective_threshold=effective_threshold,
                           confidence=confidence)
        
        # Step 6: Apply "no source, no answer" policy with adjusted threshold
        has_sufficient_confidence = confidence >= effective_threshold
        if not has_sufficient_confidence:
            logger.info("answer_rejected_low_confidence",
                       confidence=confidence,
                       threshold=effective_threshold)
            return self._build_no_answer_result(
                language=language,
                reason="low_confidence",
                retrieved_docs=retrieved_docs,
                confidence=confidence,
                debug_info=debug_info
            )
        
        # Step 7: Build successful result
        sources = self._extract_sources(answer_docs)
        
        result = RAGResult(
            answer=generated_answer,
            confidence=confidence,
            sources=sources,
            has_answer=True,
            language=language,
            intent=None,  # Can be populated by NLP module if needed
            retrieved_docs=answer_docs if (used_direct_kb_answer or used_playbook_fallback) else retrieved_docs,
            debug_info=debug_info  # Include debug info
        )
        
        logger.info("rag_answer_success",
                   confidence=confidence,
                   num_sources=len(sources),
                   has_answer=True)
        
        return result
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of input text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (defaults to "tr" for Turkish)
        """
        # Simple heuristic: check for Turkish characters
        turkish_chars = set("ğüşıöçĞÜŞİÖÇ")
        if any(char in text for char in turkish_chars):
            return "tr"
        return "en"
    
    def _build_no_answer_result(
        self,
        language: str,
        reason: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.0,
        debug_info: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """
        Build a RAGResult for cases where we cannot provide an answer.
        
        Args:
            language: Language code
            reason: Reason for no answer ("no_documents", "low_scores", etc.)
            retrieved_docs: Optional retrieved documents
            confidence: Confidence score (default: 0.0)
            debug_info: Optional debug information dictionary
            
        Returns:
            RAGResult with has_answer=False
        """
        if language == "tr":
            answer = "Mevcut kaynaklara dayanarak güvenilir bir cevap üretemiyorum."
        else:
            answer = "I cannot provide a reliable answer based on available sources."
        
        sources = self._extract_sources(retrieved_docs) if retrieved_docs else []
        
        # Use provided debug_info or collect from retrieved docs
        if debug_info is None and retrieved_docs:
            first_doc = retrieved_docs[0]
            debug_info = {
                "alpha_used": first_doc.get("alpha_used"),
                "bm25_results_count": first_doc.get("_bm25_source_count", 0),
                "embedding_results_count": first_doc.get("_embedding_source_count", 0),
                "hybrid_results_count": len(retrieved_docs),
                "query_type": None  # Not determined for no-answer cases
            }
        
        logger.debug("no_answer_result_built", 
                    reason=reason,
                    num_sources=len(sources))
        
        return RAGResult(
            answer=answer,
            confidence=confidence,
            sources=sources,
            has_answer=False,
            language=language,
            intent=None,
            retrieved_docs=retrieved_docs or [],
            debug_info=debug_info
        )
    
    def _extract_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieved documents.
        
        Args:
            docs: Retrieved documents
            
        Returns:
            List of source dictionaries with essential fields
        """
        sources = []
        seen_sources: set[tuple[str, str]] = set()
        for doc in docs:
            doc_id = doc.get("ticket_id") or doc.get("id") or doc.get("doc_id", "unknown")
            title = doc.get("short_description", doc.get("title", ""))[:100]
            dedupe_id = _doc_document_id(doc) or str(doc_id)
            if "_chunk_" in dedupe_id:
                dedupe_id = dedupe_id.split("_chunk_", 1)[0]
            source_key = (dedupe_id, str(title))
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            source = {
                "doc_id": doc_id,
                "doc_type": doc.get("doc_type", "ticket"),
                "title": title,
                "snippet": doc.get("description", doc.get("text", ""))[:360],
                "relevance_score": min(1.0, max(0.0, float(doc.get("score", 0.0))))
            }
            sources.append(source)
        
        return sources
    
    def answer_query(
        self,
        query: str,
        top_k: int = 10,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer user query using RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer, confidence, and optionally sources
            
        Process:
            1. Retrieve relevant documents
            2. Build prompt with context
            3. Generate answer (placeholder for now)
            4. Estimate confidence
            5. Apply "no source, no answer" policy
            6. Return result with sources
        """
        logger.info("rag_query_started", query=query, top_k=top_k)
        
        # 1. Retrieve relevant documents
        retrieved_docs = self.retriever.search(query, top_k=top_k)
        
        if not retrieved_docs:
            logger.warning("no_documents_retrieved", query=query)
            return self._build_no_answer_response(
                "No relevant documents found in the knowledge base.",
                []
            )
        
        # 2. Build prompt
        prompt = self.prompt_builder.build_prompt(
            query=query,
            documents=retrieved_docs,
            max_context_length=self.max_context_length
        )
        
        # 3. Generate answer
        # TODO: Implement actual LLM generation
        # For now, placeholder response
        if self.llm_model is None:
            answer = "LLM model not loaded. Please initialize the model."
            confidence = 0.0
            has_answer = False
        else:
            # Placeholder for actual generation
            answer = self._generate_answer(prompt)
            
            # 4. Estimate confidence
            retrieval_scores = [doc.get("score", 0.0) for doc in retrieved_docs]
            confidence, has_answer = self.confidence_estimator.estimate_confidence(
                answer=answer,
                query=query,
                retrieved_docs=retrieved_docs,
                retrieval_scores=retrieval_scores
            )
        
        # 5. Apply policy: if confidence too low, return explicit "I don't know"
        if not has_answer:
            logger.info("low_confidence_answer_rejected", 
                       confidence=confidence,
                       query=query)
            return self._build_no_answer_response(
                "I don't have enough information in the knowledge base to answer this question reliably.",
                retrieved_docs if return_sources else []
            )
        
        # 6. Build successful response
        response = {
            "answer": answer,
            "confidence": confidence,
            "has_answer": True,
            "num_sources": len(retrieved_docs)
        }
        
        if return_sources:
            response["sources"] = self.prompt_builder.extract_sources_from_context(retrieved_docs)
        
        logger.info("rag_query_completed",
                   query=query,
                   confidence=confidence,
                   num_sources=len(retrieved_docs))
        
        return response
    
    def _generate_answer(self, prompt: Dict[str, str]) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Prompt dictionary with system and user messages
            
        Returns:
            Generated answer
        """
        # TODO: Implement actual LLM generation
        # This is a placeholder that should be replaced with:
        # - transformers pipeline for local models
        # - API calls for hosted models
        # - Proper error handling and generation parameters
        
        logger.warning("llm_generation_not_implemented")
        return "LLM generation not yet implemented. This is a placeholder response."
    
    def _build_no_answer_response(
        self,
        message: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build response for cases where we cannot answer.
        
        Args:
            message: Message explaining why we can't answer
            retrieved_docs: Retrieved documents (may be empty)
            
        Returns:
            Response dictionary
        """
        response = {
            "answer": message,
            "confidence": 0.0,
            "has_answer": False,
            "num_sources": len(retrieved_docs)
        }
        
        if retrieved_docs:
            response["sources"] = self.prompt_builder.extract_sources_from_context(retrieved_docs)
        else:
            response["sources"] = []
        
        return response
    
    def batch_answer(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple queries in batch.
        
        Args:
            queries: List of user queries
            top_k: Number of documents to retrieve per query
            
        Returns:
            List of response dictionaries
        """
        results = []
        for query in queries:
            result = self.answer_query(query, top_k=top_k, return_sources=True)
            results.append(result)
        
        return results


