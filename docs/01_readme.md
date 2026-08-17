# Scope dan prinsip AIPulse

Dokumen ini menjelaskan batas produk yang benar-benar diimplementasikan.

## Tujuan

AIPulse menyediakan alur lokal yang dapat diuji untuk model klasifikasi tabular:

1. Membersihkan data tanpa mempelajari transformasi dari keseluruhan dataset.
2. Memisahkan training/validation secara stratified.
3. Melatih pipeline preprocessing dan estimator pada training set saja.
4. Mengukur performa dan menjalankan release gate.
5. Menyimpan artefak dengan checksum serta metadata registry.
6. Menyediakan akses registry/training melalui API ber-RBAC.
7. Menampilkan status aktual melalui UI yang menggunakan API.

## Prinsip implementasi

- **Evidence over claims.** Fairness dan compliance tidak otomatis dianggap lulus.
- **Secure failure.** Tanpa JWT secret, health menjadi degraded dan endpoint terlindungi ditolak.
- **Explicit boundaries.** UI tidak membaca registry atau file model secara langsung.
- **Reproducible behavior.** Random state, schema validation, dan semantic versioning eksplisit.
- **Recoverable writes.** Registry, snapshots, feature data, dan artefak memakai atomic replace.
- **Operational honesty.** File storage dan in-process limits didokumentasikan sebagai batas lokal.

## Bukan bagian dari baseline saat ini

- Distributed training atau inference serving.
- Kubernetes, Airflow, Kafka, Feast, MLflow, DVC CLI, dan object storage cloud.
- External identity provider, key rotation, atau multi-tenant authorization.
- Sertifikasi GDPR/EU AI Act atau jaminan immutable/WORM audit storage.
- Automated deployment ke production.

Komponen tersebut dapat ditambahkan ketika ada kebutuhan dan acceptance criteria yang jelas, bukan
sekadar sebagai dependency deklaratif.
