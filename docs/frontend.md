# The frontend

The web interface: TypeScript and React, running in the browser.

Written for someone who has not built one before. For why a frontend needs an
API at all, read [`architecture.md`](architecture.md) first.

---

## The four ideas you actually need

**HTML is structure, CSS is appearance, JavaScript is behaviour.** TypeScript is
JavaScript with type annotations — the same relationship as adding type hints to
Python, with the same payoff.

**React lets you write the page as functions.** Rather than finding an element
and updating it by hand, you write a function returning what the page *should*
look like for some data; React works out the minimal change. A *component* is
closely analogous to a function returning a matplotlib axis:

```tsx
function Metric({ label, value }: { label: string; value: number }) {
  return <div className="metric">{label}: {formatMetric(value)}</div>;
}
```

That HTML-looking syntax inside TypeScript is **JSX**. It compiles to ordinary
function calls.

**Components hold state, and changing it redraws.** `useState` gives a component
a value and a setter; calling the setter re-runs the function. That is how
typing into the prediction form updates what will be submitted.

**Data fetching is its own problem.** Every request has three possible states —
loading, failed, loaded — and forgetting one gives a blank screen. TanStack Query
handles that plus caching, so every page reads the same way:

```tsx
if (runs.isPending) return <Loading />;
if (runs.isError) return <ErrorState error={runs.error} />;
return <DataTable columns={columns} rows={runs.data} rowKey={(r) => r.run_id} />;
```

---

## Layout

```text
frontend/src/
├── main.tsx          Startup: mounts React into index.html
├── App.tsx           Routing table — which URL shows which page
├── api/
│   ├── openapi.json  Generated from the backend. Do not hand-edit.
│   ├── schema.d.ts   Generated TypeScript types. Do not hand-edit.
│   ├── client.ts     fetch wrapper; turns failures into typed errors
│   └── hooks.ts      One hook per endpoint
├── lib/
│   └── format.ts     How numbers and dates become text — decided once
├── components/       The shared vocabulary (below)
├── pages/            One per screen: Predict, Runs, RunDetail
└── styles.css        Plain CSS; restyle via the variables at the top
```

---

## The component vocabulary

Pages compose these rather than writing markup. That is what keeps three
separately-written screens feeling like one system.

| Component | Use |
| --- | --- |
| `PageHeader` | Title, secondary line, optional right-aligned actions |
| `Section` | A titled block with a consistent empty state |
| `DataTable` | Columns described as data, so they can be derived at runtime |
| `KeyValueTable` | Two-column name/value, for parameters and tags |
| `MetricGrid` | Metric tiles, formatted by magnitude |
| `ProbabilityBars` | Per-class scores; absent for regressors |
| `FeatureForm` | Inputs generated from the model signature |
| `ArtifactGallery` | Images from one artifact folder |
| `Loading` / `EmptyState` / `ErrorState` | The three states every fetch has |

### Adding a page

1. Create it in `pages/`.
2. Compose `PageHeader` and `Section`.
3. Fetch with a hook from `api/hooks.ts` and render the three states.
4. Add a route in `App.tsx`.

New CSS is normally unnecessary — the components carry their own.

---

## Staying dataset-agnostic

Four rules, each enforced by a test rather than a comment. Breaking any ties the
interface to one dataset and makes the template useless to everyone else.

**1. No feature name appears in frontend code.** `FeatureForm` renders one
control per entry returned by `GET /api/predict/schema`, choosing the widget from
the declared `kind`. Two tests render completely different schemas — the demo
dataset versus a housing dataset with text, integer, and boolean fields — and
assert the first one's fields are absent from the second.

**2. Table columns are derived, not declared.** The runs dashboard takes the
union of metric keys it receives. A test feeds it `test_rmse` and `test_r2`,
metrics this template never ships, and they render.

**3. Types are generated, never hand-written.** `make types` regenerates
`api/schema.d.ts` from the API's OpenAPI document. Rename a Pydantic field and
the frontend fails to compile at exactly the line that needs updating.

**4. "No model yet" is a first-class state.** A fresh clone gets a `503`, shown
as guidance naming the command to run — deliberately not styled as a failure,
and deliberately not retried, since asking again will not train a model.

---

## Numbers adapt to what they are

Supporting regression put two very different kinds of number in the same table.
A **bounded score** (accuracy, R², f1) sits in [0, 1]. An **error term** (RMSE,
MAE) is in the units of your target and might be 54.75 or 128,456.79. One fixed
precision cannot serve both:

| Value | `toFixed(4)` | `formatMetric` |
| --- | --- | --- |
| `0.9561` | `0.9561` | `0.9561` |
| `54.752601` | `54.7526` | `54.75` |
| `128456.7891` | `128456.7891` | `128,456.79` |
| `0.00003` | **`0.0000`** ← lost | `3.00e-5` |

`lib/format.ts` chooses from the **magnitude of the value**, never the metric's
name. Matching on names like "accuracy" would tie the interface to one project's
vocabulary — rule 2 above, applied to numbers.

The same module trims regression predictions while leaving class labels alone:
`String(152.13381958007812)` would print seventeen digits, the model's binary
precision presented as if it were confidence.

Timestamps are relative ("3 hours ago") with the absolute value in a `title`,
because recency is usually the question in a run list.

---

## Theming

`styles.css` is plain CSS with custom properties at the top. Change the values
in `:root` to restyle the whole application; a `prefers-color-scheme` block
supplies the dark variants. No framework, no component edits.

```css
:root {
  --accent: #2b6cb0;    /* links, primary buttons, focus rings */
  --surface: #ffffff;   /* cards, tables, form backgrounds */
  --text: #1b1f24;
  --radius: 8px;
}
```

Class names follow a loose `block__element--modifier` convention, so a selector
says where it applies without searching the markup.

---

## How it talks to the backend

Requests are **relative** (`/api/...`), so the browser sends them to whatever
origin served the page. In development the Vite dev server forwards them to the
API; in production one host serves both. Neither case needs a backend address
compiled into the bundle.

`vite.config.ts` also proxies `/docs`, `/redoc`, and `/openapi.json`, so the
API's interactive documentation is reachable from the same origin.

---

## Testing

```bash
make test-frontend
```

Tests use Vitest and Testing Library, and stub `fetch` rather than requiring a
running API. They query the page the way a user would — by label and role — so
they survive markup changes but catch behavioural ones.

```bash
make lint-frontend   # TypeScript, ESLint, Prettier
npm run build        # production bundle
```

TypeScript runs in strict mode, including `noUncheckedIndexedAccess`, which is
why array and record lookups are guarded.
