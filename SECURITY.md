# Security policy

## Supported version

Security fixes are applied to the current default branch. This repository is a baseline for
internal deployment and must be threat-modeled for its target environment before production use.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, personal data, or model/data
artifacts. Report privately to the repository maintainers and include affected version, impact,
reproduction steps, and a suggested mitigation when available.

## Deployment requirements

- Generate a unique high-entropy `AIPULSE_JWT_SECRET`; never commit it or reuse development values.
- Terminate TLS at a trusted reverse proxy and restrict direct access to service ports.
- Replace local HMAC JWT issuance with organizational OIDC/JWKS for multi-user deployments.
- Keep dataset, model, registry, and audit volumes access-controlled and backed up.
- Run dependency/image scanning and review the SBOM before promotion.
- Treat fairness/compliance `not_assessed` as unresolved work, never as a successful control.
