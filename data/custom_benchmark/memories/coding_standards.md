# Coding Standards

## UI Component Rules
Always use COC (Component-on-Component) pattern in all UI components. Never use FCOC (Functional Component-on-Component). This was decided after the Q2 refactor broke three production pages.

## API Conventions
All REST endpoints must use snake_case for field names. The mobile team depends on this — they auto-generate Swift models from the API schema.

## Error Handling
Never swallow exceptions silently. Every catch block must either re-raise, log at WARNING or above, or return an explicit error response. Silent failures caused the invoice bug in March.
