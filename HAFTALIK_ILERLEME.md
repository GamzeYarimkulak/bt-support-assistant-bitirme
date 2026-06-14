# Bitirme Projesi Haftalık İlerleme Raporu

## Proje Bilgileri

| Alan | Bilgi |
|------|-------|
| **Öğrenci Adı Soyadı** | Gamze Yarımkulak |
| **Öğrenci No** | 22360859036 |
| **Proje Başlığı** | Bağlamsal Farkındalıklı BT Destek Asistanı: Hibrit RAG ve Anomali Tespiti ile Güvenilir Yanıt Üretimi |
| **Danışman** | Prof. Dr. Turgay Tugay Bilgin |
| **Dönem** | 2025-2026 Bahar |

---

## İş Planı


| Hafta | Tarih Aralığı | Planlanan İş | Tahmini Tamamlanma (%) | Durum |
|-------|---------------|--------------|------------------------|-------|
| 1 | 01.04 - 05.04 | Bitirme projesi için public GitHub reposunun oluşturulması, proje kapsamının netleştirilmesi, haftalık planın hazırlanması ve temel klasör yapısının kurulması | %10 | ✅ Tamamlandı |
| 2 | 06.04 - 12.04 | Veri kaynaklarının düzenlenmesi, veri ön işleme adımlarının belirlenmesi ve anonimleştirme hattının oluşturulması | %20 | ✅ Tamamlandı |
| 3 | 13.04 - 19.04 | BM25 tabanlı kelime temelli arama yapısının eklenmesi ve örnek sorgularla ilk retrieval testlerinin yapılması | %30 | ✅ Tamamlandı |
| 4 | 27.04 - 03.05 | Embedding tabanlı semantik arama hattının geliştirilmesi ve vektör indeksleme altyapısının hazırlanması | %40 | ✅ Tamamlandı |
| 5 | 04.05 - 10.05 | Hibrit retrieval yapısının kurulması, kelime temelli ve semantik sonuçların birleştirilmesi, sıralama mantığının iyileştirilmesi | %52 | ✅ Tamamlandı |
| 6 | 11.05 - 17.05 | RAG/chat pipeline’ının geliştirilmesi, kaynaklı yanıt üretimi ve “kaynak yoksa cevap yok” mantığının eklenmesi | %65 | ✅ Tamamlandı |
| 7 | 18.05 - 24.05 | Anomali tespiti modülünün geliştirilmesi, semantik drift ve örnek anomali senaryolarının test edilmesi | %75 | ✅ Tamamlandı |
| 8 | 01.06 - 07.06 | API ve frontend entegrasyonunun yapılması, sohbet ekranı ve anomali panelinin birlikte çalışacak şekilde düzenlenmesi | %85 | ✅ Tamamlandı |
| 9 | 08.06 - 14.06 | Testlerin genişletilmesi, hata senaryolarının kontrol edilmesi, sistem performansının ve yanıt kalitesinin değerlendirilmesi | %93 | ✅ Tamamlandı |
| 10 | 15.06 - 21.06 | Dokümantasyonun tamamlanması, son hata düzeltmeleri, demo/sunum hazırlığı ve final teslim öncesi genel kontrol | %100 | ⬜ Başlamadı |

**Durum simgeleri:** ⬜ Başlamadı | 🔄 Devam Ediyor | ✅ Tamamlandı | ⚠️ Gecikti

---

## Haftalık İlerleme Kayıtları

### Hafta 9 *(Tarih: 08.06.2026 - 14.06.2026)*

**Plandaki hedef:**

- Testlerin genişletilmesi
- Hata senaryolarının kontrol edilmesi
- Sistem performansının değerlendirilmesi
- Yanıt kalitesinin ve retrieval başarımının analiz edilmesi

**Bu hafta yaptıklarım:**

- Veri hazırlama hattını düzenleyerek ham ticket ve bilgi bankası verilerinden `data/processed` çıktılarının üretilebilmesini sağladım
- Ticket ve KB dokümanlarının birlikte indekslenebilmesi için BM25 ve FAISS tabanlı index hattını güncelledim
- Büyük veri setiyle çalışmayı kolaylaştırmak için indexleme sürecine opsiyonel `--limit` parametresi ekledim
- Retrieval değerlendirme scriptini geliştirerek Recall@5, Recall@10, Precision@5 ve nDCG@10 metriklerini hesapladım
- Genel kullanıcı sorguları ile ticket-spesifik sorguların farklı değerlendirilmesi gerektiğini belirleyerek kategori ve alt kategori bazlı ek metrikler ekledim
- Retrieval değerlendirme veri setindeki tekrar eden ve genel ifadeli sorguların exact ID skorlarını düşürdüğünü analiz ettim
- Anomali tespit modülü için bağımsız validation veri seti oluşturdum
- Volume spike, category shift, semantic drift ve combined anomaly senaryolarını içeren anomali ground-truth yapısını genişlettim
- Anomali değerlendirme scriptini geliştirerek precision, recall, F1, specificity, false positive adayları ve severity uyumu gibi metrikleri hesapladım
- Anomali motorunda review candidate, warning ve critical alert ayrımını daha anlaşılır hale getirdim
- Frontend anomaly ekranına model quality paneli ekleyerek validation metriklerinin arayüzde gösterilmesini sağladım
- Frontend tasarımını daha sade, modern ve operasyon paneli görünümüne uygun olacak şekilde düzenledim
- Chat arayüzünde kaynak gösterimi, bağlam yeterlilik skoru ve Türkçe yanıt akışını kontrol ettim
- Test ortamındaki eksik bağımlılıkları kontrol ederek `pyarrow` ve `pypdf` gibi gerekli paketleri requirements dosyasına ekledim
- `python -m pytest` komutu ile testleri çalıştırarak mevcut testlerin geçtiğini doğruladım
- Bitirme raporunda kullanılmak üzere retrieval ve anomaly performans grafikleri hazırladım

**Plana göre durumum:**

- Hafta 9 hedefleri büyük ölçüde tamamlandı
- Retrieval, RAG ve anomaly modülleri için değerlendirme metrikleri üretildi
- Sistem performansı hem metrik dosyaları hem de frontend paneli üzerinden izlenebilir hale getirildi
- Testler çalıştırılarak sistemin temel bileşenlerinin beklendiği gibi davrandığı doğrulandı

**Karşılaştığım sorunlar / zorluklar:**

- Retrieval değerlendirmesinde genel sorgular için tek bir ticket ID beklemenin gerçekçi olmadığı görüldü
- Exact ID metrikleri ile kategori/alt kategori bazlı metriklerin ayrı yorumlanması gerektiği belirlendi
- Anomali değerlendirmesinde yalnızca pozitif eventler üzerinden ölçüm yapmanın yanıltıcı olabileceği fark edildi
- Bu nedenle negatif günleri de içeren bağımsız validation seti oluşturularak daha dengeli bir değerlendirme yapıldı
- Semantic drift ölçümünün güvenilir olabilmesi için ticket embedding alanlarının mevcut olması gerektiği görüldü
- Frontend ve backend arasında anomaly response alanlarının uyumlu hale getirilmesi için ek düzenlemeler yapıldı

**Gelecek hafta hedefim:**

- Final dokümantasyonu tamamlamak
- Bitirme raporunun performans değerlendirme ve sonuç bölümlerini düzenlemek
- Demo sırasında kullanılacak ekran görüntülerini ve metrikleri sabitlemek
- GitHub reposunu son proje yapısına göre temizlemek ve güncellemek
- Teslim öncesi son genel kontrolü yapmak

### Hafta 8 *(Tarih: 01.06.2026 - 07.06.2026)*

**Plandaki hedef:**
- API ve frontend entegrasyonunun yapılması
- Sohbet ekranı ve anomali panelinin birlikte çalışacak şekilde düzenlenmesi

**Bu hafta yaptıklarım:**
- FastAPI tabanlı API katmanını yapılandırdım ve router entegrasyonlarını tamamladım
- Chat servisleri için REST endpointlerini geliştirdim
- Anomali istatistikleri ve drift analizi için API endpointlerini sisteme ekledim
- Sohbet ekranı ve anomali panelini içeren web arayüzünü oluşturdum
- Frontend ile backend arasındaki veri alışverişini sağlayan entegrasyonu gerçekleştirdim
- Kullanıcı oturumu ve sohbet geçmişi yönetimi için gerekli frontend mantığını ekledim
- API çağrılarının doğruluğunu test ederek sistem bileşenlerinin birlikte çalıştığını doğruladım

**Plana göre durumum:**
- Hafta 8 hedefleri büyük ölçüde tamamlandı
- API ve frontend katmanları entegre edilerek çalışır duruma getirildi

**Karşılaştığım sorunlar / zorluklar:**
- Frontend ve backend arasındaki veri akışının senkronizasyonu için ek düzenlemeler gerekti
- Farklı endpointlerden dönen sonuçların arayüzde tutarlı gösterimi üzerinde çalışıldı

**Gelecek hafta hedefim:**
- Test kapsamını genişletmek
- Hata senaryolarını değerlendirmek
- Retrieval, RAG ve anomaly modüllerinin performansını ölçmek
- Sistemin genel yanıt kalitesini analiz etmek

### Hafta 7 *(Tarih: 18.05.2026 - 24.05.2026)*

**Plandaki hedef:**
- Anomali tespiti modülünün geliştirilmesi
- Semantik drift senaryolarının test edilmesi
- Örnek anomali durumlarının analiz edilmesi

**Bu hafta yaptıklarım:**
- Sistem davranışlarını analiz etmek için anomaly detection modülünü projeye ekledim
- Retrieval ve kullanıcı sorgularındaki olağandışı durumları tespit etmeye yönelik feature extraction yapısını geliştirdim
- Semantik drift senaryolarını değerlendirmek amacıyla drift detection mekanizmasını oluşturdum
- Anomali tespit sürecini yöneten temel engine yapısını geliştirdim
- Farklı senaryolar üzerinden anomaly detection davranışını doğrulamak amacıyla test dosyaları ve senaryo scriptleri ekledim

**Plana göre durumum:**
- Hafta 7 hedefleri büyük ölçüde tamamlandı
- Temel anomaly detection ve drift analysis altyapısı oluşturuldu

**Karşılaştığım sorunlar / zorluklar:**
- Semantik drift davranışlarının ölçülmesi için uygun feature yapılarının belirlenmesi gerekti
- Farklı anomaly senaryolarında eşik değerlerinin dengelenmesi için testler yapıldı
- Retrieval davranışlarındaki normal ve anormal örüntülerin ayrıştırılması üzerinde çalışıldı

**Gelecek hafta hedefim:**
- API ve frontend entegrasyonunu geliştirmek
- Sohbet ekranı ile anomaly panelini birlikte çalışacak hale getirmek
- Sistem bileşenlerini ortak endpoint yapısında birleştirmek
  

### Hafta 6 *(Tarih: 11.05.2026 - 17.05.2026)*

**Plandaki hedef:**
- RAG/chat pipeline’ının geliştirilmesi
- Kaynaklı yanıt üretimi
- “Kaynak yoksa cevap yok” mantığının eklenmesi

**Bu hafta yaptıklarım:**
- Retrieval sonuçlarını kullanarak yanıt üreten temel RAG pipeline yapısını projeye ekledim
- Sistem yanıtlarında kullanılacak prompt şablonlarını oluşturdum
- Yanıt güven skorlarını değerlendiren confidence kontrol mekanizmasını geliştirdim
- Kaynak bulunamayan durumlarda sistemin güvenilir olmayan cevap üretmesini engelleyecek temel kontrol yapısını ekledim

**Plana göre durumum:**
- Hafta 6 hedeflerinin büyük bölümü tamamlandı
- Temel RAG/chat pipeline yapısı oluşturuldu

**Karşılaştığım sorunlar / zorluklar:**
- Retrieval sonuçlarının prompt içine uygun şekilde yerleştirilmesi için yapı düzenlemeleri gerekti
- Güven skoru eşiklerinin belirlenmesi sırasında farklı senaryolar test edildi

**Gelecek hafta hedefim:**
- Anomali tespit modülünü geliştirmek
- Semantik drift ve örnek anomali senaryolarını test etmek


### Hafta 5 *(Tarih: 04.05.2026 - 10.05.2026)*

**Plandaki hedef:**
- Hibrit retrieval yapısının kurulması
- Kelime temelli ve semantik sonuçların birleştirilmesi
- Sıralama mantığının iyileştirilmesi

**Bu hafta yaptıklarım:**
- BM25 ve embedding tabanlı retrieval sonuçlarını birleştiren hibrit retrieval modülünü projeye ekledim
- Retrieval sonuçlarını normalize ederek tek bir hibrit skor üzerinden sıralama yapısını oluşturdum
- Sorgu özelliklerine göre BM25 ve embedding ağırlıklarını dinamik olarak ayarlayan ağırlıklandırma modülünü ekledim
- Kısa teknik sorgular, uzun açıklamalı sorgular ve dengeli sorgular için dinamik ağırlıklandırma davranışını test ettim

**Plana göre durumum:**
- Hafta 5 hedefleri büyük ölçüde tamamlandı
- Hibrit retrieval hattı kuruldu ve test edilmeye başlandı

**Karşılaştığım sorunlar / zorluklar:**
- BM25 ve embedding skorlarının farklı ölçeklerde olması nedeniyle skor normalizasyonu gerekti
- Kısa teknik sorgular ile uzun açıklamalı sorgular için aynı ağırlıklandırma yaklaşımının yeterli olmadığı görüldü

**Gelecek hafta hedefim:**
- RAG/chat pipeline yapısını oluşturmak
- Kaynaklı yanıt üretimi ve “kaynak yoksa cevap yok” mantığını sisteme eklemek

### Hafta 4 *(Tarih: 27.04.2026 - 03.05.2026)*

**Plandaki hedef:**
- Embedding tabanlı semantik arama hattının geliştirilmesi
- Vektör indeksleme altyapısının hazırlanması

**Bu hafta yaptıklarım:**
- Embedding tabanlı semantik retrieval modülünü projeye ekledim
- SentenceTransformer kullanarak metinlerin vektör temsillerini oluşturdum
- FAISS tabanlı vektör indeksleme yapısını kurarak benzerlik aramasını gerçekleştirdim
- Retrieval performansını ölçmek için precision, recall, MAP ve nDCG metriklerini içeren değerlendirme modülünü ekledim
- Semantik arama ile kelime temelli arama arasındaki farkları analiz ederek sistem davranışını gözlemledim

**Plana göre durumum:**
- Hafta 4 hedefleri büyük ölçüde tamamlandı
- Semantik retrieval altyapısı oluşturuldu ve test edilebilir hale getirildi

**Karşılaştığım sorunlar / zorluklar:**
- Embedding boyutları ve indeksleme sürecinin performansa etkisi değerlendirildi
- Vektör arama sonuçlarının anlamlılığını test etmek için uygun örnek veri oluşturulması gerekti

**Gelecek hafta hedefim:**
- Hibrit retrieval yapısını kurmak (BM25 + embedding)
- Retrieval sonuçlarını birleştirme ve sıralama mantığını geliştirmek

**Kişisel değerlendirme:**
Bu hafta öğrendiğim yöntemler model performansını artırmak açısından oldukça faydalıydı.

### Hafta 3 *(Tarih: 13.04.2026 - 19.04.2026)*

**Plandaki hedef:**
- BM25 tabanlı kelime temelli arama yapısının eklenmesi
- Örnek sorgularla ilk retrieval testlerinin yapılması

**Bu hafta yaptıklarım:**
- BM25 tabanlı kelime temelli retrieval modülünü projeye ekledim
- Retrieval modülünün proje yapısına entegrasyonunu başlattım
- Retrieval performansını kontrol etmek için temel test dosyası ekledim
- BM25 arama mantığının örnek dokümanlar ve ticket dönüşümü üzerinden çalışmasını doğruladım

**Plana göre durumum:**
- Hafta 3 hedefleri büyük ölçüde tamamlandı
- Kelime temelli retrieval hattı oluşturuldu ve test aşaması başlatıldı

**Karşılaştığım sorunlar / zorluklar:**
- Sorgu ve doküman metinlerinin uygun şekilde işlenmesi için yapı gözden geçirildi
- Test senaryolarında örnek veri ile anlamlı sonuç üretimini dengelemek gerekti

**Gelecek hafta hedefim:**
- Embedding tabanlı semantik retrieval yapısını eklemek
- Vektör indeksleme altyapısını hazırlamak

---
### Hafta 2 *(Tarih: 06.04.2026 - 12.04.2026)*

**Plandaki hedef:**
- Veri kaynaklarının düzenlenmesi
- Veri ön işleme adımlarının belirlenmesi
- Anonimleştirme hattının oluşturulması

**Bu hafta yaptıklarım:**
- Örnek ticket veri seti oluşturdum
- Veri ön işleme sürecini planladım
- E-posta, telefon, IP ve isim bilgilerini maskeleyen anonimleştirme modülünü geliştirdim
- Veri pipeline yapısına ait dokümantasyon ekledim

**Plana göre durumum:**
- Hafta 2 hedeflerine ulaşıldı
- Retrieval aşamasına geçmek için veri hazırlama zemini oluşturuldu

**Karşılaştığım sorunlar / zorluklar:**
- Farklı veri tipleri için ortak anonimleştirme yaklaşımını sade tutmak gerekti
- Gerçek veri yerine örnek veri kullanımı planlandı

**Gelecek hafta hedefim:**
- BM25 tabanlı retrieval yapısı eklenecek
- İlk arama testleri yapılacak

---

### Hafta 1 *(Tarih: 01.04.2026 - 05.04.2026)*

**Plandaki hedef:**
- Bitirme projesi için public GitHub reposunun oluşturulması
- Proje kapsamının ve hedeflerinin netleştirilmesi
- 10 haftalık iş planının hazırlanması
- Temel klasör yapısının kurulması

**Bu hafta yaptıklarım:**
- Bitirme projesi için public GitHub reposu oluşturdum
- Proje başlığını, kapsamını ve genel hedeflerini netleştirdim
- Haftalık ilerleme takibi için `HAFTALIK_ILERLEME.md` dosyasını ekledim
- Projede kullanılacak temel klasör yapısını planladım ve oluşturmaya başladım
- Geliştirme sürecinde izlenecek 10 haftalık iş planını hazırladım

**Plana göre durumum:**
- İlk hafta için belirlenen hedefler tamamlandı
- Projenin geliştirme sürecini düzenli takip edebilmek için gerekli repo ve dokümantasyon altyapısı oluşturuldu

**Karşılaştığım sorunlar / zorluklar:**
- Projenin kapsamını haftalara dengeli biçimde dağıtmak için planlama yapılması gerekti
- Kullanılacak klasör yapısı ve geliştirme sırasını netleştirme aşamasında başlangıçta karar verilmesi gereken noktalar oldu

**Gelecek hafta hedefim:**
- Veri kaynaklarını düzenlemek
- Veri ön işleme adımlarını netleştirmek
- Anonimleştirme hattının ilk sürümünü oluşturmak

---

<!--
ŞABLON: Yeni hafta eklemek için aşağıdaki bloğu kopyalayıp üste yapıştırın.

### Hafta X *(Tarih: GG.AA.YYYY - GG.AA.YYYY)*

**Plandaki hedef:**
- 

**Bu hafta yaptıklarım:**
- 

**Plana göre durumum:**
- 

**Karşılaştığım sorunlar / zorluklar:**
- 

**Gelecek hafta hedefim:**
- 

---
-->
