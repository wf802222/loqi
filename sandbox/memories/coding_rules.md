# Project Coding Rules

## Naming Convention
All function parameters and variables must use snake_case, not camelCase.
This applies to function names, parameter names, and local variables.
The codebase had inconsistent naming that caused confusion during code review.

## Error Handling
All functions that look up items by ID must raise a ValueError with a
descriptive message if the item is not found, rather than returning None.
Returning None caused silent failures in three places last month.

## Input Validation
All functions that accept user input (name, email, role) must validate
inputs before processing. Email must contain @, name must be non-empty,
role must be one of the allowed values.

## Type Hints
All function signatures must include type hints for parameters and
return values. This is required for the project's mypy strict mode.
