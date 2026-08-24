# Development workflow

HybridRAG uses short-lived feature branches, Conventional Commits, and pull requests into `main`. The repository's contributor process is defined in `CONTRIBUTING.md`.

## Set up a working copy

```bash
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Use `pip install -e ".[all]"` when the change touches optional API, UI, ingestion, evaluation, observability, or agent features. The [Tooling](tooling.md) page explains the extras and Make targets.

## Branch and implementation cycle

1. Fork the repository and update your local `main`.
2. Create a descriptive branch, for example:

   ```bash
   git checkout -b feature/native-rerank-options
   ```

3. Make one coherent change at a time.
4. Add tests under the matching `tests/` subtree.
5. Run focused tests while iterating, then the broader checks before opening a PR.
6. Update user documentation and `CHANGELOG.md` when behavior changes.

Avoid mixing filter syntaxes. `$vectorSearch` accepts MongoDB operators such as `$eq` and `$gte`; Atlas `$search` accepts operators such as `equals` and `range`. The project keeps separate builders under `src/hybridrag/enhancements/filters/`.

## Commit messages

Use Conventional Commits:

```text
feat: add a query mode
fix(search): preserve tenant filter during fusion
docs: update installation instructions
perf: reduce vector-search round trips
test: cover index synchronization timeout
```

Common types are `feat`, `fix`, `docs`, `perf`, `test`, `refactor`, `build`, and `ci`. Add a scope when it makes the affected subsystem clearer.

## Pre-PR checks

```bash
make lint
make typecheck
make test
```

For release-sensitive retrieval work, run:

```bash
make release-gate-fast
```

MongoDB-backed tests require the local Atlas-compatible service:

```bash
make mongo-up
make test-integration
```

See [Testing](testing.md) for marker selection and live-provider requirements.

## Pull requests

Push the branch to your fork and open a PR against `main`. The description should state:

- what changed and why;
- the affected public behavior;
- how the change was tested;
- any configuration, data, or migration impact.

The PR should include tests, relevant documentation, and a changelog entry. CI runs lint, formatting checks, Python 3.11 and 3.12 tests, smoke tests, and a package build as configured in `.github/workflows/ci.yml`.

Reviewers should pay special attention to async calls, tenant and ACL predicates, BSON serialization, and MongoDB search-stage syntax. See [Security](../security.md) for trust boundaries and [Design decisions](../background/design-decisions.md) for retrieval policy.
