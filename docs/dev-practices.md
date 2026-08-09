# Development practices

Why this project is set up the way it is: fast local feedback, a pre-commit
gate that mirrors CI, and CI that mirrors pre-commit. None of this is
DS/ML-specific — it applies to any Python (or TypeScript) project — but the
examples below are this repo's actual configuration, not hypothetical ones.

For the commands to run day to day, see [backend.md](backend.md#quality-gates).

---

## Linting and formatting: Ruff

[Ruff](https://docs.astral.sh/ruff/) replaces what used to be four or five
separate tools (Flake8 plus a pile of plugins, isort, pyupgrade, Black) with
one Rust binary that runs in milliseconds. That speed is not a nice-to-have —
a linter people wait on is a linter people start skipping, in editors and in
pre-commit alike.

Two things make Ruff config worth getting right early:

- **Pick a rule set deliberately, in `pyproject.toml`.** This project selects
  `E, F, I, UP, B, D` (pycodestyle, pyflakes, isort, pyupgrade, bugbear,
  pydocstyle) rather than Ruff's much larger catalogue — enough to catch real
  mistakes and keep imports/docstrings consistent, without rules that just
  generate churn. Widen it deliberately, not by accident.
- **Let it fix what it can.** `ruff --fix` and `ruff format` run in
  pre-commit so formatting arguments never reach code review — the tool has
  an opinion, and the opinion is applied automatically.

## Type checking: ty

[ty](https://docs.astral.sh/ty/) is Astral's type checker — same team as
Ruff and uv, same design goal of being fast enough that nobody has a reason
to skip it. It replaced mypy here, and the migration surfaced a lesson worth
generalizing:

**A type checker is only as strict as what it can see.** mypy's
`ignore_missing_imports = true` — a setting almost every mypy config carries,
because plenty of dependencies ship without a `py.typed` marker — silently
downgraded every untyped import to `Any`. That's not a narrow gap: it meant
`self.model.predict(...)` on a scikit-learn estimator was never actually
checked, because scikit-learn doesn't ship `py.typed`. ty checks against a
library's real source instead of demanding an opt-in marker, and it caught
the gap immediately: `BaseEstimator` itself declares neither `fit` nor
`predict` (those come from mixins that vary by concrete class), so treating
a dynamically-loaded model as `BaseEstimator` was always a lie the old setup
never noticed.

The fix is the general pattern worth keeping: when code legitimately depends
on structural, duck-typed behavior — a plugin system, a dynamically imported
class, anything selected by a config string rather than an import — define a
`typing.Protocol` naming exactly the methods relied on, instead of reaching
for the nearest concrete base class or a blanket `# type: ignore`. See
`_Estimator` in `backend/src/ml/train_model.py` for the shape of it. Where a
capability is genuinely optional (`predict_proba` on some estimators but not
others), a `hasattr` check plus a narrow `cast` at that one call site is
honest about what's actually being trusted, and confines the trust to a
single line instead of the whole module.

A few other differences worth knowing if you're used to mypy:

- ty has no `strict = true` switch — its defaults are already close to what
  mypy's strict mode has to be opted into, so there's less config to carry.
- No `.mypy_cache`-style directory to gitignore or clean up; ty's caching is
  in-memory and per-invocation.
- Config lives under `[tool.ty]` in `pyproject.toml` (or a `ty.toml`),
  including per-path rule overrides via `[[tool.ty.overrides]]` — useful for
  relaxing specific rules in, say, a fixtures directory, without disabling
  them project-wide.

## Pre-commit hooks

[pre-commit](https://pre-commit.com/) is the point where linting and type
checking stop being something CI reports on ten minutes later and become
something that fails before the commit exists. A few rules make that
actually pleasant instead of annoying:

- **Keep hooks fast.** Everything in `.pre-commit-config.yaml` here is a
  linter, formatter, or type checker — no test suite. Slow hooks get
  `--no-verify`'d, which defeats the point.
- **Pin hook revisions**, the same as any other dependency —
  `.pre-commit-config.yaml` pins `rev:` for every hook. An unpinned hook is a
  linter whose rules can change under you between one `git commit` and the
  next.
- **Keep the pinned version in sync with the dev dependency.** Ruff and ty
  are both pinned in `.pre-commit-config.yaml` *and* declared as dev
  dependencies in `backend/pyproject.toml`. If those drift — say, `uv`
  bumps `ruff` in the lockfile but the pre-commit pin stays behind — you get
  a real failure mode: code that passes `uv run ruff check` locally but
  fails the hook, or vice versa. Bump both together.
- **Run it in CI too**, with `--all-files`, not just locally. The local hook
  only ever sees staged files; a rule change or a new file that slipped in
  before the hook was added won't be caught until something runs the full
  sweep. `.github/workflows/pre-commit.yaml` does exactly that on every
  push and PR.

## CI/CD

The two workflows here split cleanly by what they're actually verifying,
which is worth calling out as the general shape to copy:

- **`tests.yaml` and `pre-commit.yaml` run on every push and PR** — they
  answer "is this change correct," and both backend and frontend are
  checked in the same run so a PR can't merge with one half broken.
- **`build.yml` only runs on `main`, and skips paths that can't affect the
  built image** (`docs/**`, `**.md`, `.github/**`) — it answers "should we
  ship this," which is a different question with a different, more
  expensive answer (building and pushing a container). Coupling that to
  every doc typo wastes CI minutes and, worse, mints a new image tag for a
  change that has no runtime effect.
- **CI installs dependencies the same way a developer would** —
  `astral-sh/setup-uv` plus `uv sync --group dev`, reading the same
  `uv.lock` a local `uv sync` would — rather than a hand-maintained
  `requirements.txt` that can silently diverge from what's actually pinned.
  If CI and local installs can produce different dependency trees, "works
  on my machine" stops being a joke.
- **A check that isn't required is decorative.** These workflows are only
  worth what branch protection makes of them — configure required status
  checks on `main` for `tests` and `pre-commit`, or a red CI run is just a
  notification nobody has to act on.

## Dependency hygiene

Worth doing on a cadence, not just when something breaks: diff the
declared dependencies in `pyproject.toml` against what's actually imported
in `src/`. Unused dependencies aren't free — they widen the install, slow
CI, and enlarge the surface a supply-chain compromise of *any* transitive
dependency can reach. (This repo carried an unused `httpx` and `httpx2` in
`backend/pyproject.toml`'s dev group for exactly this reason — nothing
imported either — before this review removed them.) A stray unfamiliar
package name is also worth a second look on its own merits before assuming
it's a mistake: it's cheap to check who publishes it and why, and not every
surprising name is a problem.

Commit the lockfile (`uv.lock`) and pin the interpreter (`.python-version`)
for the same reason: reproducibility should not depend on whatever happens
to be installed on a given machine that day, and CI should be resolving the
exact same dependency graph a local `uv sync` produces.

---

The frontend applies the same shape with different tools — ESLint and
Prettier instead of Ruff, `tsc --noEmit` instead of ty — and the same
principles hold: fast checks, pinned versions, and a CI job that runs
exactly what a developer runs locally. See [frontend.md](frontend.md).
