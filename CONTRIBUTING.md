# Contributing to sandx-embed

Thanks for your interest. This document covers how to set up a development environment, run tests, and submit a pull request.

---

## Development setup

```bash
git clone https://github.com/sandxlab/sandx-embed
cd sandx-embed
pip install -e ".[dev]"
```

For HNSW index support:

```bash
pip install usearch
```

For encoder support:

```bash
pip install sentence-transformers
```

## Running tests

```bash
pytest tests/ -q
```

Tests that require optional dependencies are skipped automatically if those packages are not installed.

With coverage:

```bash
pytest tests/ --cov=sandx_embed --cov-report=term-missing -q
```

## Linting

```bash
ruff check src tests
```

We use `ruff` with `line-length = 100`. Fix lint errors before opening a PR — CI will reject anything that fails.

## Code style

- Type-annotate all public functions and methods.
- No comments explaining *what* code does. Only add a comment when the *why* is non-obvious (a hidden constraint, a workaround, a subtle invariant).
- Optional heavy dependencies (`sentence-transformers`, `usearch`) must be imported lazily inside functions, never at module level.
- `encode()` must always return a `float32` numpy array of shape `(N, D)`.

## Before opening a PR

1. Tests pass: `pytest tests/ -q`
2. Lint passes: `ruff check src tests`
3. New behaviour has test coverage.
4. Optional dependencies remain optional — the base package must import cleanly with only `numpy` installed.

## Pull request process

- Branch off `main`. Name your branch `feat/short-description` or `fix/short-description`.
- Keep PRs focused. One logical change per PR.
- PR description should explain *why*, not just *what*.
- At least one approving review is required before merge.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/sandxlab/sandx-embed/issues).

- **Bug reports:** include Python version, sandx-embed version (`pip show sandx-embed`), minimal reproducing example, and the full traceback.
- **Feature requests:** describe the use case, not just the feature. What problem does it solve?

## Design principles

sandx-embed is shared embedding infrastructure used by all SandX engines. It has no opinion about what you embed — text, records, graph nodes. Changes that make the encoder or index more opinionated about domain will not be merged.

---

Apache 2.0 license. By contributing you agree your changes are released under the same license.
