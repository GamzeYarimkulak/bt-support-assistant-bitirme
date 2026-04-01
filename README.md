#  Bağlamsal Farkındalıklı BT Destek Asistanı  
**Hibrit RAG ve Anomali Tespiti ile Güvenilir Yanıt Üretimi**

Bu proje, kurumsal Bilgi Teknolojileri (BT) destek süreçlerinde kullanılan kayıtlar üzerinden **doğru, güvenilir ve izlenebilir yanıtlar üreten** ve aynı zamanda **anomalileri erken tespit edebilen** yapay zekâ destekli bir sistem geliştirmeyi amaçlamaktadır.

Proje, TÜBİTAK 2209-B kapsamında yürütülen bir araştırma çalışmasıdır.

---

##  Problem Tanımı

Kurumsal BT destek sistemlerinde:

- Benzer sorunlar farklı şekilde yazıldığı için eşleşememektedir  
- Yanıtların hangi kaynağa dayandığı çoğu zaman belirsizdir  
- Tekrarlayan sorunlar erken fark edilememektedir  
- Türkçe ve teknik dil içeren kayıtlar klasik sistemler için zorludur  

Bu durum:
- Çözüm sürelerini uzatır  
- Yanlış yönlendirmelere neden olur  
- Operasyonel verimliliği düşürür  

---

##  Projenin Amacı

Bu proje ile:

-  Doğru bilgiye dayalı yanıt üretimi  
-  Kaynak gösteren (traceable) cevaplar  
-  Anomali ve trend değişimlerini erken tespit  
-  Daha hızlı çözüm süreçleri  

hedeflenmektedir.

Sistem, yalnızca cevap üretmekle kalmaz, aynı zamanda **proaktif bir karar destek mekanizması** sunar.

---

##  Sistem Mimarisi

Proje iki ana bileşenden oluşmaktadır:

### 1️ Hibrit RAG (Retrieval-Augmented Generation)

- BM25 (kelime temelli arama)
- Embedding tabanlı semantik arama
- Hibrit skor birleştirme (fusion)
- Kaynağa dayalı yanıt üretimi

 “**Kaynak yoksa cevap yok**” prensibi uygulanır.

Bu sayede:
- Halüsinasyon (uydurma bilgi) azaltılır  
- Yanıtların güvenilirliği artar  

---

### 2️ Anomali Tespit Modülü

- Zaman pencereleri üzerinden veri analizi  
- Embedding tabanlı semantik drift tespiti  
- Kümelenme (k-means / DBSCAN)  
- Erken uyarı mekanizması  

Sistem:
- Yeni konu oluşumlarını  
- Ani artışları  
- Beklenmeyen değişimleri  

tespit ederek erken uyarı üretir.

---

##  Sistem Akışı

Projenin genel akışı şu şekildedir:

1. Veri toplama ve anonimleştirme  
2. Veri ön işleme ve indeksleme  
3. Hibrit arama (BM25 + embedding)  
4. Sonuçların birleştirilmesi  
5. LLM ile yanıt üretimi  
6. Kaynak ve güven skoru sunumu  
7. Anomali tespiti ve raporlama  

Bu yapı, hem doğru yanıt üretimini hem de sistem izlenebilirliğini sağlar. 

---

##  Kullanılacak Teknolojiler

- Python  
- FastAPI  
- BM25 (rank_bm25)  
- Sentence Transformers / Embedding modelleri  
- FAISS / benzeri vektör indeksleme  
- OpenAI / LLM API (opsiyonel)  
- Pandas / NumPy  
- Scikit-learn  

---

##  Hedeflenen Başarı Kriterleri

Proje kapsamında hedeflenen metrikler:

- Retrieval doğruluğu (nDCG@10 ≥ 0.75)  
- Kaynaklı yanıt oranı ≥ %70  
- Anomali tespiti precision ≥ %80  
- Ortalama yanıt süresi < 2 saniye  
- Tekrarlayan kayıt oranında azalma  

Bu metrikler sistemin hem teknik hem operasyonel başarısını ölçmek için kullanılacaktır. 

---

##  Veri Güvenliği

- Tüm veriler anonimleştirilir  
- KVKK uyumlu veri işleme uygulanır  
- Gerçek kullanıcı verileri paylaşılmaz  

---

## 📁 Proje Yapısı

```bash
.
├── app/                # API ve endpointler
├── core/               # Retrieval, RAG ve model bileşenleri
├── data_pipeline/      # Veri işleme ve anonimleştirme
├── frontend/           # Web arayüzü
├── tests/              # Testler
├── docs/               # Dokümantasyon
├── README.md
└── HAFTALIK_ILERLEME.md
