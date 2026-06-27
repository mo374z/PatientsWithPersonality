# CLAUDE.md

## Code Style

- Concise, readable, and maintainable above all else
- No inline comments — code should be self-explanatory through naming and structure
- When something can be shortened or generalized, do it
- Prefer generic, reusable components over one-off implementations
- Mirror the existing project structure and coding style at all times

## Defaults & Configuration

- Do not set default values for parameters that affect experiment behavior
- If a required config is missing, raise an explicit error — never silently fall back
- The user is responsible for correct configuration; the code is responsible for failing loudly when it is wrong

## Structure & Refactoring

- Before adding new code, check if existing components can be extended or generalized
- Flatten unnecessary abstractions — fewer layers is better if clarity is preserved
- Keep functions and modules focused; if something grows, split it along natural boundaries

## What to Avoid

- No clever one-liners that sacrifice readability
- No defensive defaults that mask misconfiguration
- No inline comments — rename or restructure instead
- No duplication — extract shared logic immediately
