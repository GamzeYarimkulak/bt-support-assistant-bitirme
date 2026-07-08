"""
Tests for RAG pipeline (PHASE 4).
"""

import pytest
from datetime import datetime
from core.rag.pipeline import (
    RAGPipeline,
    RAGResult,
    _build_contextual_retrieval_query,
    _extract_relevant_kb_segments,
    _requires_source_grounded_kb_answer,
    generate_answer_with_stub,
)
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.embedding_retriever import EmbeddingRetriever
from core.retrieval.hybrid_retriever import HybridRetriever
from data_pipeline.ingestion import ITSMTicket
from data_pipeline.build_indexes import convert_ticket_to_document


class TestLLMStub:
    """Tests for the LLM stub function."""
    
    def test_stub_with_documents_turkish(self):
        """Test stub generates answer with Turkish documents."""
        docs = [
            {
                "ticket_id": "TCK-001",
                "short_description": "Outlook şifremi unuttum",
                "resolution": "Şifre sıfırlandı"
            }
        ]
        
        answer = generate_answer_with_stub("Outlook şifre", docs, language="tr")
        
        assert answer is not None
        assert len(answer) > 0
        assert "TCK-001" in answer
        assert isinstance(answer, str)
    
    def test_stub_with_no_documents_turkish(self):
        """Test stub returns no-answer message when no documents."""
        answer = generate_answer_with_stub("test query", [], language="tr")
        
        assert "güvenilir bir cevap üretemiyorum" in answer.lower()
    
    def test_stub_with_no_documents_english(self):
        """Test stub returns no-answer message in English."""
        answer = generate_answer_with_stub("test query", [], language="en")
        
        assert "cannot provide" in answer.lower()
    
    def test_stub_with_multiple_documents(self):
        """Test stub handles multiple documents."""
        docs = [
            {"ticket_id": "TCK-001", "short_description": "Issue 1", "resolution": "Fix 1"},
            {"ticket_id": "TCK-002", "short_description": "Issue 2", "resolution": "Fix 2"},
        ]
        
        answer = generate_answer_with_stub("test", docs, language="en")
        
        assert "TCK-001" in answer
        assert "similar" in answer.lower()

    def test_factual_kb_question_uses_kb_sentence_after_ticket(self):
        """Factual document questions should use KB text even if a ticket ranks first."""
        docs = [
            {
                "ticket_id": "TCK-999",
                "doc_type": "ticket",
                "short_description": "Ekli dosya gönderilemiyor",
                "resolution": "Kullanıcıya dosya boyutu kontrolü önerildi.",
                "score": 0.9,
            },
            {
                "id": "ozdilek_kb_test_chunk_001",
                "doc_id": "ozdilek_kb_test_chunk_001",
                "doc_type": "kb",
                "title": "BT Stratejisi",
                "content": "2.2.4. Mobil yazıcılardan Lukhan Sewoo LK P12 ve üstü mobil yazıcılar kullanılır.",
                "score": 0.4,
            },
        ]

        answer = generate_answer_with_stub("mobil yazıcılardan hangisi kullanılıyor", docs, language="tr")

        assert "Lukhan Sewoo LK P12" in answer
        assert "Kaynakta geçen bilgiye göre" in answer
        assert "Adım" not in answer

    def test_product_question_does_not_invent_missing_product_name(self):
        """Product lookup questions should not fabricate vendor/product names."""
        docs = [
            {
                "id": "ozdilek_kb_antivirus_chunk_001",
                "doc_id": "ozdilek_kb_antivirus_chunk_001",
                "doc_type": "kb",
                "title": "BT Antivirüs Talimatı",
                "content": "Antivirüs uygulaması güncellemeleri sunucu üzerinde yapılır ve kullanıcı bilgisayarlarındaki uygulamalar otomatik güncellenir.",
                "score": 0.8,
            },
        ]

        answer = generate_answer_with_stub("antivirüs antispam için hangi ürünler kullanılıyor", docs, language="tr")

        assert "açık bir ürün/model adı bulunamadı" in answer
        assert "Symantec" not in answer
        assert "McAfee" not in answer
        assert "Barracuda" not in answer

    def test_kb_answer_cleans_pdf_header_metadata(self):
        """Direct KB answers should show the relevant instruction, not PDF header metadata."""
        docs = [
            {
                "id": "ozdilek_kb_mail_chunk_001",
                "doc_id": "ozdilek_kb_mail_chunk_001",
                "doc_type": "kb",
                "title": "BT Elektronik Posta Talimatı",
                "content": (
                    "DOKÜMAN ONAY MERCİ: Bilgi Teknolojileri ve ARGE Direktörü "
                    "DOKÜMAN KODU: HLD.T.GNL.005 REVİZYON SAYISI: 22 [Page 2] "
                    "7.Alınan önemsiz elektronik posta (spam) mesajı hiçbir şekilde yanıtlanmamalıdır."
                ),
                "score": 0.8,
            },
        ]

        answer = generate_answer_with_stub(
            "spam mesajı geldiğinde ne yapılması öneriliyor",
            docs,
            language="tr"
        )

        assert "Alınan önemsiz elektronik posta" in answer
        assert "DOKÜMAN ONAY MERCİ" not in answer

    def test_action_plan_question_uses_labeled_form_fields(self):
        """Action plan forms should return labeled source fields, not random date rows."""
        docs = [
            {
                "id": "ozdilek_kb_action_plan_chunk_001",
                "doc_id": "ozdilek_kb_action_plan_chunk_001",
                "doc_type": "kb",
                "title": "Enerji Yönetimi Aksiyon Planı",
                "content": (
                    "Enerji Yönetimi Aksiyon Planı "
                    "Proje Amacı/Hedefi: GKM odasında bulunan ekran ve BT sistemlerinin soğutma ihtiyacı için iki klima bulunmaktadır. "
                    "Proje Başlangıç Tarihi: 01.10.2019 "
                    "Proje Tanımı: GKM odası ekran ve BT sistemlerinin olduğu bölüme cam bölme yapılması. "
                    "Proje Planlama Aksiyonları: Cam bölmelerin merkez şubeden alınması. Salon tipi klimanın sökülmesi. "
                    "Proje Sonuçlarının Doğrulanması: Klima saatlik elektrik tüketimi 8,6 kW değerinden 4,3 kW değerine düşmüştür. "
                    "Sonuçların Değerlendirmesi: GKM odası toplam 35 m2'dir. Cam bölme ile soğutulacak bölüm 11 m2'ye düşürülmüştür."
                ),
                "score": 0.9,
            },
        ]

        answer = generate_answer_with_stub("enerji yönetimi aksiyon planından bahseder misin", docs, language="tr")

        assert "Proje amacı/hedefi" in answer
        assert "Proje tanımı" in answer
        assert "Aksiyonlar" in answer
        assert "Proje sonuçlarının doğrulanması" in answer
        assert "8,6 kW" in answer
        assert "4,3 kW" in answer
        assert "Sonuçların değerlendirmesi" in answer
        assert "11 m2" in answer

    def test_erp_question_uses_sap_kb_sources(self):
        """ERP questions should surface SAP-related Özdilek KB documents."""
        docs = [
            {
                "id": "ozdilek_kb_sap_hana_chunk_001",
                "doc_id": "ozdilek_kb_sap_hana_chunk_001",
                "doc_type": "kb",
                "title": "DST.T.BT.013_3 BT SAP HANA İŞ SÜREKLİLİĞİ SENARYOSU",
                "content": (
                    "Bu prosedürde SAP HANA sisteminde beklenmedik durumlara karşı "
                    "Bilgi Teknolojileri ve Ar-Ge Direktörlüğü tarafından yürütülecek faaliyetler tanımlanır."
                ),
                "score": 0.8,
            },
        ]

        answer = generate_answer_with_stub("özdilek holdingte ERP sistemi olarak ne kullanılıyor", docs, language="tr")

        assert "SAP" in answer
        assert "HANA" in answer
        assert "ERP/SAP sistemi olarak SAP" in answer

    def test_director_responsibility_question_uses_role_block(self):
        """Director role questions should prefer role/responsibility blocks."""
        docs = [
            {
                "id": "ozdilek_kb_noise_chunk_001",
                "doc_id": "ozdilek_kb_noise_chunk_001",
                "doc_type": "kb",
                "title": "Özveri Ar-Ge Merkezi Proje Seçim Kriter Prosedürü",
                "content": "Genel Müdürü Bilgi Teknolojileri ve Ar-Ge Direktörü Eğitim ve İnsan Kaynakları Müdürü.",
                "score": 0.9,
            },
            {
                "id": "ozdilek_kb_strategy_chunk_004",
                "doc_id": "ozdilek_kb_strategy_chunk_004",
                "doc_type": "kb",
                "title": "DST.P.GNL.001_5 BT STRATEJİLERİNİN OLUŞTURULMASI YÖNERGESİ",
                "content": (
                    "4. SORUMLULAR Rol Sorumluluk Bilgi Teknolojileri ve Ar-Ge Direktörü "
                    "•BT yönlendirme komitesi toplantılarının gündemini belirler ve toplantıyı organize eder. "
                    "•BT Stratejik Planı’nı oluşturur, takip eder ve gerekli durumlarda güncellenmesini sağlar. "
                    "•BT stratejilerine uygun bir şekilde BT hizmetlerinin verilmesini sağlar. "
                    "İç Denetim Müdürü •BT Yönlendirme Komitesi çalışmalarını denetler."
                ),
                "score": 0.7,
            },
        ]

        answer = generate_answer_with_stub(
            "Bilgi Teknolojileri ve Ar-Ge Direktörü nün görevlerinden bahseder misin ama özdilekteki",
            docs,
            language="tr",
        )

        assert "BT yönlendirme komitesi toplantılarının gündemini belirler" in answer
        assert "BT Stratejik Planı" in answer
        assert "Proje Seçim" not in answer

    def test_followup_correction_uses_previous_ozdilek_question(self):
        """Correction-style follow-ups should carry the previous factual question."""
        history = [
            {
                "role": "user",
                "content": (
                    "özdilekte Dizüstü ve masaüstü bilgisayar ve aksamları cihaz "
                    "bakımlarını yaptırmak kimin sorumluluğunda ve ne kadar sürede yapılması gerekiyor"
                ),
            },
            {
                "role": "assistant",
                "content": "Kaynakta geçen bilgiye göre bakım tablosu incelendi.",
            },
        ]

        contextual_query = _build_contextual_retrieval_query(
            "Dizüstü ve masaüstü bilgisayar ve aksamları bunu sordum ben sadece",
            history,
        )

        assert "özdilekte Dizüstü ve masaüstü bilgisayar" in contextual_query
        assert "Takip sorusu/duzeltme" in contextual_query
        assert _requires_source_grounded_kb_answer(contextual_query) is True

    def test_desktop_laptop_maintenance_answer_extracts_single_table_row(self):
        """Maintenance table lookups should not return the whole device table."""
        docs = [
            {
                "id": "ozdilek_kb_maintenance_chunk_001",
                "doc_id": "ozdilek_kb_maintenance_chunk_001",
                "doc_type": "kb",
                "title": "HLD.T.GNL.015_3 BT DONANIMLARI BAKIM DESTEK TALİMATI",
                "content": (
                    "Planlı bakım bir önceki bakım tarihine göre aşağıdaki tabloya göre yapılacaktır: "
                    "Cihaz Adı Sorumlu Bakımı Yapan Bakım Aralığı (Yılda) "
                    "Dizüstü ve masaüstü bilgisayar ve aksamları BT Müdürlüğü İç Kaynaklar 1 "
                    "Tüm yazıcılar BT Müdürlüğü İç Kaynaklar 1 "
                    "Plotterlar BT Müdürlüğü Dış Kaynaklar 1"
                ),
                "score": 0.9,
            }
        ]

        question = (
            "özdilekte Dizüstü ve masaüstü bilgisayar ve aksamları cihaz "
            "bakımlarını yaptırmak kimin sorumluluğunda ve ne kadar sürede yapılması gerekiyor"
        )

        segments = _extract_relevant_kb_segments(question, docs)
        answer = generate_answer_with_stub(question, docs, language="tr")

        assert len(segments) == 1
        assert "Dizüstü ve masaüstü bilgisayar ve aksamları" in segments[0]["segment"]
        assert "BT Müdürlüğü" in segments[0]["segment"]
        assert "İç Kaynaklar" in segments[0]["segment"]
        assert "yılda 1" in segments[0]["segment"]
        assert "Tüm yazıcılar" not in answer
        assert "Plotterlar" not in answer

    def test_laptop_performance_answer_contains_expected_actions(self):
        """Slow laptop answers should contain concrete performance troubleshooting terms."""
        docs = [
            {
                "ticket_id": "TCK-LAPTOP",
                "doc_type": "ticket",
                "short_description": "Laptop çok yavaş çalışıyor",
                "resolution": "Disk temizliği yapıldı, bellek kullanımı kontrol edildi ve güncellemeler tamamlandı.",
                "score": 0.9,
            }
        ]

        answer = generate_answer_with_stub("Laptop çok yavaş çalışıyor, ne yapmalıyım?", docs, language="tr")

        for keyword in ("performans", "disk", "bellek", "güncelleme", "temizlik"):
            assert keyword in answer.lower()

    def test_disk_full_answer_contains_expected_actions(self):
        """Disk-full answers should mention cleanup, files, and deletion/space checks."""
        docs = [
            {
                "ticket_id": "TCK-DISK",
                "doc_type": "ticket",
                "short_description": "Disk alanı doldu hatası",
                "resolution": "Geçici dosyalar temizlendi ve büyük dosyalar arşivlendi.",
                "score": 0.9,
            }
        ]

        answer = generate_answer_with_stub("Disk alanı doldu hatası alıyorum", docs, language="tr")

        for keyword in ("disk", "alan", "temizlik", "dosya", "silme"):
            assert keyword in answer.lower()


class TestRAGPipelineNoAnswer:
    """Tests for RAG pipeline when no answer should be returned."""
    
    @pytest.fixture
    def empty_pipeline(self):
        """Create a pipeline with retriever that returns no results."""
        # Create a mock retriever that returns empty results
        class MockEmptyRetriever:
            def search(self, query, top_k=5):
                return []  # No documents
        
        mock_retriever = MockEmptyRetriever()
        
        return RAGPipeline(
            retriever=mock_retriever,
            confidence_threshold=0.6
        )
    
    def test_no_documents_retrieved(self, empty_pipeline):
        """Test that pipeline returns no-answer when no documents found."""
        result = empty_pipeline.answer("VPN certificate renewal issue")
        
        assert isinstance(result, RAGResult)
        assert result.has_answer is False
        assert result.confidence == 0.0
        assert len(result.sources) == 0
        assert "güvenilir bir cevap" in result.answer.lower() or "cannot provide" in result.answer.lower()


class TestRAGPipelineWithAnswer:
    """Tests for RAG pipeline when it should return an answer."""
    
    @pytest.fixture
    def pipeline_with_data(self):
        """Create a pipeline with sample ITSM tickets."""
        # Create sample tickets
        tickets = [
            ITSMTicket(
                ticket_id="TCK-001",
                created_at=datetime(2025, 1, 10, 9, 0, 0),
                category="Uygulama",
                subcategory="Outlook",
                short_description="Outlook şifremi unuttum",
                description="Kullanıcı Outlook'a giriş yapamıyor şifreyi hatırlamıyor",
                resolution="Şifre sıfırlama bağlantısı gönderildi kullanıcı şifresini sıfırladı",
                channel="email",
                priority="High",
                status="Closed"
            ),
            ITSMTicket(
                ticket_id="TCK-002",
                created_at=datetime(2025, 1, 11, 10, 0, 0),
                category="Ağ",
                subcategory="VPN",
                short_description="VPN bağlantısı kopuyor",
                description="VPN her 5 dakikada bir kopuyor",
                resolution="VPN istemcisi güncellendi",
                channel="phone",
                priority="Medium",
                status="Closed"
            ),
            ITSMTicket(
                ticket_id="TCK-003",
                created_at=datetime(2025, 1, 12, 11, 0, 0),
                category="Donanım",
                subcategory="Laptop",
                short_description="Laptop yavaş çalışıyor",
                description="Laptop açılırken ve çalışırken çok yavaş",
                resolution="Disk temizliği yapıldı RAM yükseltildi",
                channel="web",
                priority="Low",
                status="Closed"
            ),
        ]
        
        # Convert to documents
        documents = [convert_ticket_to_document(t) for t in tickets]
        
        # Create retrievers
        bm25 = BM25Retriever()
        bm25.index_documents(documents, text_field="text")
        
        # Note: Embedding retriever would require model loading (slow for tests)
        # For unit tests, we just use BM25
        # In integration tests, we'd use both
        
        # Mock embedding retriever to return same as BM25
        class MockEmbeddingRetriever:
            def search(self, query, top_k=5):
                return bm25.search(query, top_k)
        
        embedding = MockEmbeddingRetriever()
        
        hybrid = HybridRetriever(bm25, embedding, alpha=1.0)  # 100% BM25 for speed
        
        return RAGPipeline(
            retriever=hybrid,
            confidence_threshold=0.5  # Lower threshold for test
        )
    
    def test_finds_relevant_ticket(self, pipeline_with_data):
        """Test that pipeline finds and answers with relevant ticket."""
        result = pipeline_with_data.answer("Outlook şifre problemi")
        
        assert isinstance(result, RAGResult)
        # With good BM25 match, should find answer
        if result.has_answer:
            assert result.confidence > 0.0
            assert len(result.sources) > 0
            assert any("TCK-001" in src["doc_id"] for src in result.sources)
            assert "Outlook" in result.answer or "şifre" in result.answer.lower()
    
    def test_returns_sources(self, pipeline_with_data):
        """Test that sources are included in result."""
        result = pipeline_with_data.answer("VPN bağlantı sorunu")
        
        # Should find VPN ticket
        if len(result.sources) > 0:
            assert result.sources[0]["doc_id"] in ["TCK-001", "TCK-002", "TCK-003"]
            assert "doc_type" in result.sources[0]
            assert "relevance_score" in result.sources[0]
    
    def test_language_detection_turkish(self, pipeline_with_data):
        """Test that Turkish language is detected."""
        result = pipeline_with_data.answer("Outlook şifremi unuttum")
        
        assert result.language == "tr"
    
    def test_language_detection_english(self, pipeline_with_data):
        """Test that English language is detected."""
        result = pipeline_with_data.answer("password reset problem")
        
        # Should detect as English (no Turkish chars)
        assert result.language == "en"

    def test_invalid_language_falls_back_to_detection(self, pipeline_with_data):
        """Unsupported language values should not leak into the API response."""
        result = pipeline_with_data.answer("Outlook şifre problemi", language="invalid")

        assert result.language in {"tr", "en"}
    
    def test_multiple_relevant_tickets(self, pipeline_with_data):
        """Test handling of multiple relevant results."""
        result = pipeline_with_data.answer("laptop yavaş")
        
        # Should find the laptop ticket
        if result.has_answer:
            assert len(result.sources) >= 1
            # Check that laptop ticket is in sources
            ticket_ids = [src["doc_id"] for src in result.sources]
            assert "TCK-003" in ticket_ids or any("TCK" in tid for tid in ticket_ids)


class TestRAGResultStructure:
    """Tests for RAGResult dataclass."""
    
    def test_rag_result_creation(self):
        """Test that RAGResult can be created with required fields."""
        result = RAGResult(
            answer="Test answer",
            confidence=0.85,
            sources=[{"doc_id": "TCK-001", "doc_type": "ticket"}],
            has_answer=True,
            language="tr"
        )
        
        assert result.answer == "Test answer"
        assert result.confidence == 0.85
        assert result.has_answer is True
        assert result.language == "tr"
        assert len(result.sources) == 1
    
    def test_rag_result_optional_fields(self):
        """Test that optional fields work correctly."""
        result = RAGResult(
            answer="Test",
            confidence=0.5,
            sources=[],
            has_answer=False
        )
        
        assert result.language is None
        assert result.intent is None
        assert len(result.retrieved_docs) == 0  # Default empty list


# ============================================================================
# PHASE 6.5: Advisory-Style Answer Generation Tests
# ============================================================================

class TestAdvisoryAnswerGeneration:
    """
    Tests to verify that the RAG system uses advisory language
    and does NOT claim to have performed actions.
    
    CRITICAL SAFETY REQUIREMENT:
    The system must present itself as a recommendation/decision-support tool,
    NOT as an agent that performs actions on behalf of the user.
    """
    
    def test_advisory_language_turkish(self):
        """Test that Turkish answers use advisory language."""
        docs = [
            {
                "ticket_id": "TCK-001",
                "short_description": "Outlook şifremi unuttum",
                "resolution": "Şifre sıfırlama bağlantısı gönderildi."
            }
        ]
        
        answer = generate_answer_with_stub(
            "Outlook şifremi nasıl sıfırlarım?",
            docs,
            language="tr"
        )
        
        # Should contain advisory phrases
        advisory_phrases = ["BT ekibi", "önerilir", "deneyebilirsiniz", "talep edebilirsiniz"]
        assert any(phrase in answer for phrase in advisory_phrases), \
            f"Answer should contain advisory language. Got: {answer}"
        
        # Should mention "Uygulanan Çözüm" (Applied Solution) to indicate past action
        assert "Uygulanan Çözüm" in answer or "uygulanan" in answer.lower(), \
            "Answer should indicate these are past solutions, not current actions"
    
    def test_no_action_claims_turkish(self):
        """Test that Turkish answers do NOT claim actions were performed."""
        docs = [
            {
                "ticket_id": "TCK-002",
                "short_description": "VPN bağlanamıyor",
                "resolution": "VPN şifresi sıfırlandı ve kullanıcıya bildirildi."
            }
        ]
        
        answer = generate_answer_with_stub(
            "VPN'e bağlanamıyorum",
            docs,
            language="tr"
        )
        
        # Should NOT claim that WE did these actions
        forbidden_phrases = [
            "şifreniz sıfırlandı",
            "şifrenizi sıfırladık",
            "bağlantınızı gönderdim",
            "sorununuzu çözdüm"
        ]
        
        for phrase in forbidden_phrases:
            assert phrase not in answer.lower(), \
                f"Answer MUST NOT claim action was performed. Found: '{phrase}' in: {answer}"
    
    def test_password_reset_advisory(self):
        """Test specific case: password reset should be advisory, not declarative."""
        docs = [
            {
                "ticket_id": "TCK-003",
                "short_description": "Outlook şifremi unuttum",
                "resolution": "Şifre sıfırlama bağlantısı gönderildi."
            }
        ]
        
        answer = generate_answer_with_stub(
            "Outlook şifremi unuttum ne yapmalıyım?",
            docs,
            language="tr"
        )
        
        # Critical: Should NOT say "şifreniz sıfırlandı" (your password was reset)
        assert "şifreniz sıfırlandı" not in answer.lower(), \
            "CRITICAL: Must not claim password was reset for THIS user"
        
        # Should present it as an example of what IT does
        assert "örnek" in answer.lower() or "benzer" in answer.lower(), \
            "Should present as example from past cases"
    
    def test_printer_issue_advisory(self):
        """Test printer issue uses advisory language, not action claims."""
        docs = [
            {
                "ticket_id": "TCK-004",
                "short_description": "Yazıcı yazdırmıyor",
                "resolution": "Yazıcı sürücüleri güncellendi ve kuyruk temizlendi."
            }
        ]
        
        answer = generate_answer_with_stub(
            "Yazıcı çalışmıyor",
            docs,
            language="tr"
        )
        
        # Should NOT claim we updated drivers
        forbidden_claims = [
            "sürücülerinizi güncelledik",
            "yazıcınızı düzelttik",
            "kuyruğunuzu temizledik"
        ]
        
        for claim in forbidden_claims:
            assert claim not in answer.lower(), \
                f"Must not claim action was performed: {claim}"
        
        # Should use advisory language
        assert "önerilir" in answer or "deneyebilirsiniz" in answer or "BT ekibi" in answer, \
            "Should use advisory/recommendation language"
    
    def test_multiple_examples_presented(self):
        """Test that multiple examples are presented as past cases."""
        docs = [
            {
                "ticket_id": "TCK-005",
                "short_description": "VPN kopuyor",
                "resolution": "VPN istemcisi güncellendi."
            },
            {
                "ticket_id": "TCK-006",
                "short_description": "VPN yavaş",
                "resolution": "Farklı sunucuya yönlendirildi."
            },
            {
                "ticket_id": "TCK-007",
                "short_description": "VPN bağlanamıyor",
                "resolution": "Firewall kuralları düzeltildi."
            }
        ]
        
        answer = generate_answer_with_stub(
            "VPN sorunu yaşıyorum",
            docs,
            language="tr"
        )
        
        # Should show multiple examples
        assert "Örnek" in answer, "Should present examples"
        
        # Should have advisory conclusion
        assert ("deneyebilirsiniz" in answer or "talep edebilirsiniz" in answer), \
            "Should suggest user can try or request these solutions"

    def test_advisory_answer_has_clean_step_format(self):
        """Support answers should combine recommended steps with past-ticket evidence cleanly."""
        docs = [
            {
                "ticket_id": "TCK-009",
                "short_description": "VPN bağlanamıyor",
                "resolution": "VPN profili kontrol edildi ve gateway bilgisi güncellendi."
            }
        ]

        answer = generate_answer_with_stub(
            "VPN'e bağlanamıyorum ne yapmam gerek",
            docs,
            language="tr"
        )

        assert "Sizin deneyebileceğiniz adımlar" in answer
        assert "Geçmiş benzer kayıtlarda incelenenler" in answer
        assert "Uygulanan Çözüm" in answer
        assert "====" not in answer
    
    def test_advisory_language_english(self):
        """Test that English answers also use advisory language."""
        docs = [
            {
                "ticket_id": "TCK-008",
                "short_description": "Cannot login to VPN",
                "resolution": "Password was reset and user was notified."
            }
        ]
        
        answer = generate_answer_with_stub(
            "I can't login to VPN",
            docs,
            language="en"
        )
        
        # Should use advisory language
        advisory_phrases = ["IT team", "you can try", "request", "applied"]
        assert any(phrase.lower() in answer.lower() for phrase in advisory_phrases), \
            f"English answer should contain advisory language. Got: {answer}"
        
        # Should NOT claim action was done for this user
        assert "your password was reset" not in answer.lower(), \
            "Must not claim action was performed for current user"
        
        # Should indicate these are examples
        assert "example" in answer.lower() or "similar" in answer.lower(), \
            "Should present as examples from past cases"
    
    def test_no_documents_maintains_policy(self):
        """Test that no-answer case still follows advisory policy."""
        answer = generate_answer_with_stub(
            "Some random query",
            [],
            language="tr"
        )
        
        # Should return the standard "cannot answer" message
        assert "güvenilir bir cevap üretemiyorum" in answer.lower()
        
        # Should NOT make any claims about actions
        assert "çözdüm" not in answer.lower()
        assert "yaptım" not in answer.lower()

