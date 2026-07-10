"""
IT relevance checker - Filters out non-IT related queries.
"""

import re
from typing import Tuple
import structlog

logger = structlog.get_logger()

# IT-related keywords (Turkish and English)
IT_KEYWORDS = [
    # Hardware
    'bilgisayar', 'computer', 'laptop', 'yazıcı', 'yazıcılar', 'yazıcılardan',
    'mobil yazıcı', 'printer', 'monitör', 'monitor', 'terminal', 'terminaller',
    'el terminali', 'klavye', 'keyboard', 'mouse', 'fare', 'telefon', 'phone', 'tablet',
    'donanım', 'hardware', 'disk', 'ram', 'bellek', 'memory', 'ekran', 'screen',
    'şarj', 'charge', 'charger', 'batarya', 'battery', 'power', 'güç',
    
    # Software/Applications
    'yazılım', 'software', 'program', 'uygulama', 'app', 'outlook', 'excel', 'word',
    'teams', 'office', 'windows', 'linux', 'macos', 'sistem', 'sistemi', 'system',
    'bt', 'bilgi teknolojileri', 'sap', 'hana', 'erp', 'fiori', 'ecr', 'pos',
    
    # Network/Internet
    'ağ', 'network', 'internet', 'wifi', 'wi-fi', 'vpn', 'bağlantı', 'connection',
    'ip', 'dns', 'sunucu', 'server', 'email', 'e-posta', 'mail', 'switch', 'log',
    
    # Security/Authentication
    'şifre', 'password', 'güvenlik', 'security', 'kimlik', 'authentication', 'login',
    'giriş', 'mfa', '2fa', 'çok faktörlü', 'multi-factor', 'antivirüs',
    'antivirus', 'antispam', 'anti-spam', 'spam',
    
    # IT Support Terms
    'hata', 'error', 'sorun', 'problem', 'bug', 'çözüm', 'solution', 'destek', 'support',
    'ticket', 'kayıt', 'troubleshoot', 'giderme', 'sıfırlama', 'reset',
    'açılmıyor', 'açılmıyor', 'bozuldu', 'broken', 'çalışmıyor', 'not working',
    'mavi ekran', 'blue screen', 'bsod', 'ekran', 'screen',
    # Note: "açamıyorum" is NOT an IT keyword - it's about physical objects (bottles, etc.)
    
    # Technical Terms
    'sürücü', 'driver', 'güncelleme', 'update', 'yükleme', 'install', 'kurulum',
    'ayar', 'setting', 'yapılandırma', 'configuration', 'backup', 'yedekleme',
    'yedek', 'felaket', 'senaryo', 'surekliligi', 'sürekliliği', 'prosedür',
    'procedure', 'talimat', 'instruction', 'doküman', 'document',
    'aygıt', 'device', 'yöneticisi', 'manager', 'administrator',

    # Ozdilek knowledge-base / corporate IT document terms
    'özdilek', 'ozdilek', 'aksiyon planı', 'aksiyon plani', 'enerji yönetimi',
    'enerji yonetimi', 'sistem odası', 'sistem odasi', 'gkm', 'cam bölme',
    'cam bolme',
]

# Non-IT keywords that should be rejected (only if no IT keywords present)
# Note: These are checked first, but if IT keywords are found, query is accepted
NON_IT_KEYWORDS = [
    'şişe', 'bottle', 'kapak', 'lid',  # 'açmak' removed - too generic, conflicts with "açılmıyor"
    'yemek', 'food', 'içecek', 'drink', 'su', 'water',
    'ev', 'home', 'araba', 'car', 'yol', 'road',
    'sağlık', 'health', 'ilaç', 'medicine', 'doktor', 'doctor',
    'hava', 'hava durumu', 'weather', 'forecast', 'meteoroloji',
    'hap', 'pill', 'tablet ilaç',  # Medicine-related
]


CORPORATE_KB_ENTITY_MARKERS = (
    "ozdilek",
    "talimat",
    "dokuman",
    "belge",
    "prosedur",
    "yonerge",
    "gomulu",
    "elektronik",
    "yapay zeka",
    "corewish",
    "sap",
)

CORPORATE_KB_INTENT_MARKERS = (
    "bakim",
    "bahseder",
    "bilgi",
    "gorev",
    "ilke",
    "kullanim",
    "nasil",
    "soruml",
    "standart",
    "tasarim",
    "yetkilendirme",
    "yonetim",
    "yazilim",
)


def _normalize_keyword_text(text: str) -> str:
    translation = str.maketrans(
        "\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc",
        "cgiosuCGIOSU",
    )
    normalized = str(text or "").translate(translation).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _looks_like_corporate_kb_query(query: str) -> bool:
    normalized = _normalize_keyword_text(query)
    has_entity = any(marker in normalized for marker in CORPORATE_KB_ENTITY_MARKERS)
    has_intent = any(marker in normalized for marker in CORPORATE_KB_INTENT_MARKERS)
    return has_entity and has_intent


class ITRelevanceChecker:
    """
    Checks if a query is IT-related or not.
    """
    
    def __init__(self):
        """Initialize IT relevance checker."""
        # Compile regex patterns for faster matching
        # Use word boundaries to avoid partial matches (e.g., "açılmıyor" shouldn't match "açamıyorum")
        # For Turkish words with suffixes, word boundaries still work (e.g., "şişe" matches "şişenin")
        # But we need to be careful: "açılmıyor" should NOT match "açamıyorum"
        self.it_patterns = [re.compile(r'\b' + re.escape(kw.lower()) + r'\b', re.IGNORECASE) 
                           for kw in IT_KEYWORDS]
        self.non_it_patterns = [re.compile(r'\b' + re.escape(kw.lower()) + r'\b', re.IGNORECASE) 
                               for kw in NON_IT_KEYWORDS]
        
        logger.info("it_relevance_checker_initialized",
                   it_keywords_count=len(IT_KEYWORDS),
                   non_it_keywords_count=len(NON_IT_KEYWORDS))
    
    def is_it_related(self, query: str) -> Tuple[bool, float]:
        """
        Check if query is IT-related.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (is_it_related: bool, confidence: float)
            confidence is between 0.0 and 1.0
        """
        if not query or len(query.strip()) < 3:
            return False, 0.0
        
        query_lower = query.lower().strip()
        
        # Check for thank you messages or acknowledgments (don't reject these)
        thank_you_patterns = [
            r'^(teşekkür|thanks|thank you|tamam|ok|okay|anladım|tamamdır)',
            r'^(tamam|ok|okay).*(teşekkür|thanks)',
        ]
        for pattern in thank_you_patterns:
            if re.match(pattern, query_lower):
                # Thank you messages are neutral - don't reject but also don't process as IT query
                return False, 0.3  # Low confidence, won't be rejected
        
        # Special case: "açamıyorum" (can't open physical objects) vs "açılmıyor" (IT: won't open/start)
        # "açamıyorum" should be rejected even if it contains "aç" substring
        if "açamıyorum" in query_lower or "açamıyor" in query_lower:
            # Check if it's about physical objects (bottle, door, etc.) vs IT (app, program)
            physical_keywords = ["şişe", "bottle", "kapak", "lid", "kapı", "door", "kutu", "box"]
            if any(kw in query_lower for kw in physical_keywords):
                # It's about physical objects, not IT
                logger.debug("non_it_query_detected_physical_open", query=query[:50])
                return False, 0.95  # Very high confidence it's not IT-related
        
        if _looks_like_corporate_kb_query(query):
            return True, 0.85

        # Count IT keyword matches FIRST
        it_matches = sum(1 for pattern in self.it_patterns if pattern.search(query_lower))
        
        # If IT keywords found, query is IT-related (even if non-IT keywords also present)
        if it_matches > 0:
            if it_matches >= 2:
                return True, 0.9  # Multiple IT keywords - definitely IT-related
            else:
                return True, 0.6  # Single IT keyword - probably IT-related
        
        # Only check non-IT keywords if NO IT keywords found
        for pattern in self.non_it_patterns:
            if pattern.search(query_lower):
                logger.debug("non_it_query_detected", query=query[:50])
                return False, 0.9  # High confidence it's not IT-related
        
        # No IT keywords and no explicit non-IT keywords - likely not IT-related
        return False, 0.7
    
    def should_reject_query(self, query: str) -> bool:
        """
        Determine if query should be rejected as non-IT.
        
        Args:
            query: User query text
            
        Returns:
            True if query should be rejected, False otherwise
        """
        is_it, confidence = self.is_it_related(query)
        
        # Reject if not IT-related with high confidence
        if not is_it and confidence >= 0.7:
            logger.info("query_rejected_non_it", 
                       query=query[:50],
                       confidence=confidence)
            return True
        
        return False

