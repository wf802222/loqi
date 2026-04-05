# Deployment Rules

## Merge Freeze
No non-critical merges to main after Thursday each sprint. The mobile team cuts their release branch on Friday morning.

## Database Migrations
All migrations must be backwards-compatible. Never drop a column in the same release that stops using it — wait one release cycle. The billing outage in January was caused by a premature column drop.

## Feature Flags
New features touching payments must be behind a feature flag. No exceptions. The compliance team requires a 24-hour bake period before full rollout.
