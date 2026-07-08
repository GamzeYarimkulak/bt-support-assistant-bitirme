"""
Tests for NLP module (preprocessing and intent classification).
"""

import pytest
from core.nlp.preprocessing import TextPreprocessor
from core.nlp.intent import IntentClassifier, QueryIntent
from core.nlp.it_relevance import ITRelevanceChecker


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        processor = TextPreprocessor(lowercase=True)
        
        text = "  Hello   World!  "
        normalized = processor.normalize(text)
        
        assert normalized == "hello world!"
        assert len(normalized.split()) == 2
    
    def test_tokenize(self):
        """Test tokenization."""
        processor = TextPreprocessor()
        
        text = "How to reset password?"
        tokens = processor.tokenize(text)
        
        assert len(tokens) > 0
        assert "how" in tokens or "How" in tokens
    
    def test_clean_query(self):
        """Test query cleaning."""
        processor = TextPreprocessor()
        
        query = "  How to reset password?  "
        cleaned = processor.clean_query(query)
        
        assert cleaned == "How to reset password"
        assert not cleaned.endswith("?")


class TestIntentClassifier:
    """Tests for IntentClassifier."""
    
    def test_how_to_intent(self):
        """Test 'how to' intent detection."""
        classifier = IntentClassifier()
        
        query = "How to reset my password?"
        intent = classifier.classify(query)
        
        assert intent == QueryIntent.HOW_TO
    
    def test_troubleshoot_intent(self):
        """Test troubleshoot intent detection."""
        classifier = IntentClassifier()
        
        query = "My application is not working"
        intent = classifier.classify(query)
        
        assert intent == QueryIntent.TROUBLESHOOT
    
    def test_what_is_intent(self):
        """Test 'what is' intent detection."""
        classifier = IntentClassifier()
        
        query = "What is VPN?"
        intent = classifier.classify(query)
        
        assert intent == QueryIntent.WHAT_IS
    
    def test_classify_with_confidence(self):
        """Test classification with confidence scores."""
        classifier = IntentClassifier()
        
        query = "How to fix error?"
        scores = classifier.classify_with_confidence(query)
        
        assert isinstance(scores, dict)
        assert QueryIntent.HOW_TO.value in scores
        assert QueryIntent.TROUBLESHOOT.value in scores


class TestITRelevanceChecker:
    """Tests for IT relevance filtering."""

    def test_weather_query_is_rejected(self):
        checker = ITRelevanceChecker()

        is_it, confidence = checker.is_it_related("yarın hava durumu nasıl")

        assert is_it is False
        assert confidence >= 0.8
        assert checker.should_reject_query("yarın hava durumu nasıl") is True

    def test_it_query_with_weather_word_can_still_pass(self):
        checker = ITRelevanceChecker()

        is_it, _ = checker.is_it_related("hava durumu uygulaması açılmıyor")

        assert is_it is True
        assert checker.should_reject_query("hava durumu uygulaması açılmıyor") is False

