# Project Policies

## Database Choice
This project uses PostgreSQL for all data storage. Never suggest or use
MongoDB, Redis for primary storage, or SQLite in production. Redis is
acceptable only for caching. This was decided after the MongoDB migration
disaster in Q1 that caused two weeks of downtime.

## Deployment Windows
Never deploy on Thursday or Friday. The mobile team cuts their release
branch on Friday morning and any backend instability blocks the mobile
release. All deploys must happen Monday through Wednesday.

## Payment Feature Flags
Any code that touches payment processing, billing, or Stripe integration
must be behind a feature flag. No exceptions. The compliance team requires
a 24-hour bake period before full rollout. Use the LaunchDarkly SDK for
all feature flags.

## API Versioning
All public API changes must be backwards-compatible. If a breaking change
is unavoidable, it must be released under a new API version (v2, v3, etc).
Never modify existing endpoint response schemas in place. The mobile app
pins to specific API versions and will break silently if schemas change.

## Security Review
Any PR that touches authentication, authorization, user data, or payment
logic requires a security review from a senior engineer before merge.
Do not merge security-sensitive changes without explicit approval in the
PR comments.

## Error Reporting
All caught exceptions must be reported to Sentry with the full context
(user ID, request path, relevant entity IDs). Never catch-and-swallow.
The March silent failure incident was caused by an unreported exception
in the billing pipeline.
