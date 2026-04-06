"""
Data anonymization for removing PII from ITSM tickets and documents.
"""

from typing import List, Dict, Any, Set
import re
import hashlib
import structlog

logger = structlog.get_logger()


class DataAnonymizer:
    """
    Anonymizes sensitive data in tickets and documents.
    Removes or masks: emails, phone numbers, IP addresses, names, etc.
    """
    
    # Regex patterns for PII detection
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Updated to support both 7-digit (555-1234) and 10-digit (555-123-4567) phone numbers
    PHONE_PATTERN = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}(?:[-.\s]?\d{4})?\b'
    IP_PATTERN = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    URL_PATTERN = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    # Common PII field names
    PII_FIELDS = {
        "email", "phone", "mobile", "telephone", "ip_address",
        "user_email", "requester_email", "assignee_email",
        "first_name", "last_name", "full_name", "name"
    }
    
    def __init__(self, anonymization_enabled: bool = True, hash_salt: str = "bt_support_2024"):
        """
        Initialize data anonymizer.
        
        Args:
            anonymization_enabled: Whether to enable anonymization
            hash_salt: Salt for hashing identifiers
        """
        self.enabled = anonymization_enabled
        self.hash_salt = hash_salt
        self.name_mapping: Dict[str, str] = {}  # Original -> anonymized name mapping
        
        logger.info("data_anonymizer_initialized", enabled=anonymization_enabled)
    
    def anonymize_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize a single ticket.
        
        Args:
            ticket: Ticket dictionary
            
        Returns:
            Anonymized ticket
        """
        if not self.enabled:
            return ticket
        
        anonymized = ticket.copy()
        
        # Anonymize text fields
        text_fields = ["title", "description", "resolution", "comments"]
        for field in text_fields:
            if field in anonymized and anonymized[field]:
                anonymized[field] = self.anonymize_text(anonymized[field])
        
        # Anonymize or hash PII fields
        for field in self.PII_FIELDS:
            if field in anonymized and anonymized[field]:
                if "email" in field.lower():
                    anonymized[field] = self._hash_identifier(anonymized[field])
                elif "name" in field.lower():
                    anonymized[field] = self._anonymize_name(anonymized[field])
                else:
                    anonymized[field] = self._mask_value(anonymized[field])
        
        logger.debug("ticket_anonymized", ticket_id=ticket.get("id"))
        
        return anonymized
    
    def anonymize_tickets(self, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize multiple tickets.
        
        Args:
            tickets: List of tickets
            
        Returns:
            List of anonymized tickets
        """
        if not self.enabled:
            return tickets
        
        anonymized_tickets = [self.anonymize_ticket(ticket) for ticket in tickets]
        
        logger.info("tickets_anonymized", num_tickets=len(tickets))
        
        return anonymized_tickets
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonymize PII in free-form text.
        
        Args:
            text: Input text
            
        Returns:
            Anonymized text
        """
        if not text or not self.enabled:
            return text
        
        # IMPORTANT: Order matters! Replace IP addresses BEFORE phone numbers
        # to avoid IP patterns being caught by phone regex
        
        # Replace IP addresses (first)
        text = re.sub(self.IP_PATTERN, '[IP_ADDRESS]', text)
        
        # Replace emails
        text = re.sub(self.EMAIL_PATTERN, '[EMAIL]', text)
        
        # Replace phone numbers (after IPs)
        text = re.sub(self.PHONE_PATTERN, '[PHONE]', text)
        
        # Replace URLs (keep domain anonymized)
        text = re.sub(self.URL_PATTERN, '[URL]', text)
        
        return text
    
    def _anonymize_name(self, name: str) -> str:
        """
        Anonymize a person's name consistently.
        
        Args:
            name: Original name
            
        Returns:
            Anonymized name (e.g., "User_ABC123")
        """
        if name in self.name_mapping:
            return self.name_mapping[name]
        
        # Generate consistent pseudonym
        name_hash = hashlib.sha256(f"{name}{self.hash_salt}".encode()).hexdigest()[:8]
        pseudonym = f"User_{name_hash.upper()}"
        
        self.name_mapping[name] = pseudonym
        return pseudonym
    
    def _hash_identifier(self, identifier: str) -> str:
        """
        Hash an identifier (email, username, etc.) for consistency.
        
        Args:
            identifier: Original identifier
            
        Returns:
            Hashed identifier
        """
        id_hash = hashlib.sha256(f"{identifier}{self.hash_salt}".encode()).hexdigest()[:16]
        return f"HASHED_{id_hash}"
    
    def _mask_value(self, value: str) -> str:
        """
        Mask a sensitive value.
        
        Args:
            value: Original value
            
        Returns:
            Masked value
        """
        if not value:
            return value
        
        # Keep first and last character, mask middle
        if len(value) <= 2:
            return "*" * len(value)
        elif len(value) <= 4:
            return value[0] + "*" * (len(value) - 2) + value[-1]
        else:
            return value[0] + "*" * (len(value) - 2) + value[-1]
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text without anonymizing.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping PII type to detected instances
        """
        detected = {
            "emails": re.findall(self.EMAIL_PATTERN, text),
            "phones": re.findall(self.PHONE_PATTERN, text),
            "ips": re.findall(self.IP_PATTERN, text),
            "urls": re.findall(self.URL_PATTERN, text),
        }
        
        # Remove empty categories
        detected = {k: v for k, v in detected.items() if v}
        
        if detected:
            logger.info("pii_detected", types=list(detected.keys()))
        
        return detected
    
    def validate_anonymization(self, original: str, anonymized: str) -> bool:
        """
        Validate that anonymization was successful.
        
        Args:
            original: Original text
            anonymized: Anonymized text
            
        Returns:
            True if no obvious PII remains
        """
        detected = self.detect_pii(anonymized)
        
        if detected:
            logger.warning("anonymization_validation_failed",
                         remaining_pii=list(detected.keys()))
            return False
        
        return True


# ============================================================================
# Standalone Functions for ITSMTicket Anonymization (PHASE 2)
# ============================================================================

# Regex patterns for PII detection (reused from DataAnonymizer)
_EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
_PHONE_PATTERN = r'\b(?:\+?90[-.\s]?)?(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}(?:[-.\s]?\d{4})?\b'
_IP_PATTERN = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
# Simple name pattern: Two or more capitalized words (e.g., "Ahmet Yılmaz")
# Matches Turkish characters like İ, Ş, Ğ, etc.
_NAME_PATTERN = r'\b[A-ZİŞĞÜÖÇ][a-zğüşöçı]+(?:\s+[A-ZİŞĞÜÖÇ][a-zğüşöçı]+)+\b'


def anonymize_text(text: str) -> str:
    """
    Anonymize PII in free-form text using token replacement.
    
    This is a pure function that detects and replaces common PII patterns:
    - Email addresses     → [EMAIL]
    - Phone numbers       → [PHONE]
    - IPv4 addresses      → [IP]
    - Person names        → [NAME] (basic heuristic: capitalized words)
    
    The function preserves:
    - Non-PII content
    - Turkish characters (ş, ğ, ı, ü, ö, ç, etc.)
    - Text structure and spacing
    
    Args:
        text: Input text that may contain PII
        
    Returns:
        Text with PII replaced by tokens
        
    Example:
        >>> anonymize_text("Email: ahmet@example.com, phone: 0555 123 4567")
        "Email: [EMAIL], phone: [PHONE]"
        
        >>> anonymize_text("Kullanıcı Ahmet Yılmaz şifresini unutmuş")
        "Kullanıcı [NAME] şifresini unutmuş"
    """
    if not text:
        return text
    
    # Order matters! Replace in this sequence to avoid conflicts:
    # 1. IP addresses (before phone numbers to avoid "192.168" being caught as phone)
    # 2. Email addresses
    # 3. Phone numbers
    # 4. Names (last, as they're most ambiguous)
    
    # Replace IP addresses
    text = re.sub(_IP_PATTERN, '[IP]', text)
    
    # Replace email addresses
    text = re.sub(_EMAIL_PATTERN, '[EMAIL]', text)
    
    # Replace phone numbers (supports Turkish +90 format and others)
    text = re.sub(_PHONE_PATTERN, '[PHONE]', text)
    
    # Replace simple person names (e.g., "Ahmet Yılmaz")
    # Note: This is a basic heuristic and may have false positives
    text = re.sub(_NAME_PATTERN, '[NAME]', text)
    
    return text


def anonymize_ticket(ticket: 'ITSMTicket') -> 'ITSMTicket':
    """
    Anonymize PII in a single ITSM ticket.
    
    This function creates a new ITSMTicket with anonymized text fields
    while preserving all other fields (ticket_id, timestamps, categories, etc.).
    
    Fields that are anonymized:
    - short_description
    - description
    - resolution
    
    Fields that are NOT changed:
    - ticket_id
    - created_at
    - category, subcategory
    - channel, priority, status
    
    Args:
        ticket: Original ITSMTicket object
        
    Returns:
        New ITSMTicket object with anonymized text fields
        
    Example:
        >>> original = ITSMTicket(
        ...     ticket_id="TCK-001",
        ...     created_at="2025-01-10 09:00:00",
        ...     short_description="User ahmet@example.com cannot login"
        ... )
        >>> anonymized = anonymize_ticket(original)
        >>> print(anonymized.short_description)
        "User [EMAIL] cannot login"
    """
    from data_pipeline.ingestion import ITSMTicket
    
    # Create a new ticket with anonymized text fields
    # Use model_dump() to get dict, then modify and create new instance
    ticket_dict = ticket.model_dump()
    
    # Anonymize only the text fields
    if ticket_dict.get('short_description'):
        ticket_dict['short_description'] = anonymize_text(ticket_dict['short_description'])
    
    if ticket_dict.get('description'):
        ticket_dict['description'] = anonymize_text(ticket_dict['description'])
    
    if ticket_dict.get('resolution'):
        ticket_dict['resolution'] = anonymize_text(ticket_dict['resolution'])
    
    # Return new immutable ticket object
    anonymized_ticket = ITSMTicket(**ticket_dict)
    
    logger.debug("ticket_anonymized", 
                ticket_id=ticket.ticket_id,
                original_short_desc=ticket.short_description[:50],
                anonymized_short_desc=anonymized_ticket.short_description[:50])
    
    return anonymized_ticket


def anonymize_tickets(tickets: List['ITSMTicket']) -> List['ITSMTicket']:
    """
    Anonymize a list of ITSM tickets.
    
    This is a convenience function that applies anonymize_ticket to each
    ticket in the list.
    
    Args:
        tickets: List of ITSMTicket objects
        
    Returns:
        List of anonymized ITSMTicket objects
        
    Example:
        >>> tickets = load_itsm_tickets_from_csv("data/tickets.csv")
        >>> anonymized = anonymize_tickets(tickets)
        >>> print(f"Anonymized {len(anonymized)} tickets")
    """
    anonymized = [anonymize_ticket(ticket) for ticket in tickets]
    
    logger.info("tickets_anonymized_batch", 
               total_tickets=len(tickets),
               anonymized_count=len(anonymized))
    
    return anonymized

