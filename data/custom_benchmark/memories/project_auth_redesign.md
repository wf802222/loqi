# Auth System Redesign

## Background
We're replacing the old session-token auth middleware because legal flagged it for storing tokens in a way that doesn't meet the new SOC2 compliance requirements. This is compliance-driven, not tech-debt cleanup.

## Constraints
- Must support existing OAuth2 flows during transition
- Session tokens must be encrypted at rest (AES-256)
- Token rotation must happen every 24 hours
- Old middleware must remain functional until all clients migrate (estimated 6 weeks from March 1)

## Architecture Decision
Using JWT with short-lived access tokens (15 min) and rotating refresh tokens. The refresh token store is PostgreSQL, not Redis, because we need audit trails for compliance.
