# Tech stack aktual

## Runtime

- Python 3.11–3.14.
- FastAPI dan Uvicorn untuk HTTP API.
- Pydantic untuk validasi request.
- PyJWT untuk verifikasi JWT HMAC pada baseline lokal.
- Streamlit untuk operations dashboard dan admin console.

## Data dan machine learning

- Pandas dan PyArrow untuk data tabular serta Parquet.
- NumPy untuk perhitungan numerik.
- scikit-learn untuk preprocessing, logistic regression, random forest, dan metrik.
- Joblib untuk serialisasi model pipeline.

## Persistence lokal

- File Parquet untuk processed data dan feature store.
- File Joblib untuk model artifact.
- Registry JSON dengan lock proses dan atomic replace.
- JSONL audit trail dengan hash chain; HMAC digunakan saat secret tersedia.
- Snapshot file dengan manifest checksum sebagai versioning lokal yang jujur.

Persistence ini sesuai untuk satu host/satu writer. Database dan object storage diperlukan untuk
high availability, multi-instance consistency, access policy terpusat, dan disaster recovery.

## Quality dan delivery

- Pytest untuk unit dan API integration tests.
- Ruff untuk lint dan format check.
- Mypy strict untuk source project; namespace eksternal tanpa typed marker dikecualikan secara sempit.
- Docker/Compose dengan non-root user, health checks, localhost port binding, dan resource limits.
- GitHub Actions menjalankan lint, format check, strict type-check, tests, dan image build.

## Dependency policy

Range dependency dibatasi pada major version. NumPy memakai environment marker karena Python 3.14
memerlukan wheel seri yang lebih baru, sementara target Python 3.11 menggunakan seri kompatibel.
Update dependency harus melewati seluruh quality gate dan review perubahan transitive dependency.
