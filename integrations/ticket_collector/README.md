# Ticket Collector Starter

Bu klasor, kurumun mevcut ticket sistemine dogrudan baglanmadan calisan
guvenli bir ilk veri toplama paketidir. Amac, ticket sisteminden alinabilen
CSV export dosyalarini ortak semaya donusturmek, yaygin kisisel veri
oruntulerini maskelemek ve RAG/anomali calismasi icin kalite raporu uretmektir.

Bu arac:

- Canli ticket sistemine yazmaz.
- Ticket kapatmaz, silmez veya degistirmez.
- Sadece disari aktarilmis CSV dosyalarini okur.
- E-posta, telefon, IPv4 ve T.C. kimlik numarasi gibi yaygin PII alanlarini
  metin iceriginde maskeler.
- Standart CSV, JSONL ve kalite raporu uretir.

## Neden Bu Paket Var?

Hazir gecmis ticket verisi yoksa veya kapanis notlari duzenli tutulmuyorsa,
once veri toplama standardi olusturmak gerekir. Bu paket, hangi ticket sistemi
kullanildigini bilmeden ilk asamada calisabilir. Kurum ServiceNow, Jira,
ManageEngine, CoreWish veya ozel bir sistem kullaniyor olabilir; sistem
bilgisi netlesince bu klasore o sisteme ozel bir API adapter eklenebilir.

## Istenen Minimum Alanlar

Minimum calisma icin:

```text
ticket_id
created_at
status
category
short_description
description
```

RAG yanit kalitesi icin cok onemli alanlar:

```text
resolution_code
action_taken
root_cause
resolution_status
```

Anomali tespiti icin onemli alanlar:

```text
created_at
category
subcategory
affected_service
priority
severity
```

Excel export alinabiliyorsa ilk asamada dosya CSV olarak kaydedilip bu aracla
islenebilir. API entegrasyonu ise ticket sistemi netlestikten sonra eklenmelidir.

## Ornek Kullanim

Standart kolon adlari olan bir CSV icin:

```powershell
python integrations\ticket_collector\run_export.py `
  --input integrations\ticket_collector\sample_ticket_template.csv `
  --output-dir tmp\ticket_collector_demo `
  --source-system sample_export `
  --hash-ticket-id
```

Kolon adlari farkliysa `config.example.yaml` dosyasini kopyalayip
`column_mapping` alanlarini kaynak export dosyasindaki kolon adlariyla
eslestirin:

```powershell
python integrations\ticket_collector\run_export.py `
  --input C:\ticket_exports\weekly_export.csv `
  --output-dir C:\ticket_exports\standardized `
  --config integrations\ticket_collector\config.example.yaml `
  --source-system corewish `
  --hash-ticket-id
```

Uretilen dosyalar:

```text
tickets_standardized_YYYYMMDD_HHMMSS.csv
tickets_standardized_YYYYMMDD_HHMMSS.jsonl
export_quality_report_YYYYMMDD_HHMMSS.json
```

## Kapanis Notu Onerisi

BT ekibinin her ticket icin uzun aciklama yazmasi beklenmemelidir. Ancak
kapanista su dort kisa alanin tutulmasi RAG icin cok degerlidir:

```text
resolution_code: VPN_PROFILE_RESET
action_taken: VPN profili yeniden olusturuldu ve baglanti test edildi.
root_cause: Eski VPN profilinde yapilandirma bozulmasi.
resolution_status: resolved
```

Ornek cozum kodlari:

```text
VPN_PROFILE_RESET
PASSWORD_RESET
MFA_REENROLL
OUTLOOK_PROFILE_REBUILD
PRINTER_DRIVER_REINSTALL
EXCHANGE_QUOTA_CHECK
SAP_AUTHORIZATION_UPDATE
ERP_PERMISSION_UPDATE
ESCALATED_TO_NETWORK_TEAM
NO_ACTION_USER_GUIDED
```

## Sonraki Adim

Bu paket CSV export ile calisir. Kurumun ticket sistemi ve API yetenekleri
netlestikten sonra `adapters/` altina sisteme ozel sadece-okuma adapter'i
eklenebilir. Bu adapter, ayni standart semaya veri uretmelidir.
