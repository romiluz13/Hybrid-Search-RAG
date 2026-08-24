# Contributing

Contributions should preserve HybridRAG's async API, MongoDB retrieval semantics, and tenant constraints. Start with the workflow below, then use the focused guides for testing, debugging, conventions, and tools.

## Contribution guides

| Page | Use it for |
| --- | --- |
| [Development workflow](development-workflow.md) | Branches, commits, pull requests, and the definition of done |
| [Testing](testing.md) | Test layout, markers, commands, fixtures, and test-writing patterns |
| [Debugging](debugging.md) | Logs, focused test runs, MongoDB pipeline inspection, and common failure isolation |
| [Patterns and conventions](patterns-and-conventions.md) | Python style, async behavior, filters, timestamps, and public API practices |
| [Tooling](tooling.md) | Packaging, formatters, linters, pre-commit, Make targets, and CI |

## Before changing code

1. Read `CONTRIBUTING.md` and the closest implementation and test files.
2. Check the relevant ADRs under `docs/adr/`, especially for storage or retrieval changes.
3. Add or update a regression test for behavior changes.
4. Keep public changes backward compatible or document migration behavior.

The repository is a library with optional API, UI, ingestion, evaluation, and agent integrations. Install only the extras needed for the change, or use `pip install -e ".[all]"` for the full contributor environment.

## Definition of done

A change is ready for review when:

- targeted and relevant suite tests pass;
- Ruff, Black/isort, and applicable type checks have been run;
- public functions have type hints and Google-style docstrings;
- user-visible changes update documentation and `CHANGELOG.md`;
- MongoDB filter syntax is correct for the selected search stage;
- no secrets, local `.env` files, or generated artifacts are included.

See [Development workflow](development-workflow.md) for the complete branch-to-PR cycle.
