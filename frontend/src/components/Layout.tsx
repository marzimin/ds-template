/**
 * Page shell: navigation and a persistent model-status indicator.
 *
 * The status line answers "is the backend up, and is a model loaded?" on every
 * page, so a blank panel is never ambiguous.
 */
import { NavLink, Outlet } from 'react-router';

import { useHealth, useReloadModel } from '../api/hooks';

function ModelStatus() {
  const health = useHealth();
  const reload = useReloadModel();

  if (health.isPending) return <span className="status status--idle">checking…</span>;

  if (health.isError) {
    return (
      <span className="status status--down" title={String(health.error)}>
        API unreachable
      </span>
    );
  }

  if (!health.data.model_available) {
    return <span className="status status--warn">no model trained</span>;
  }

  return (
    <span className="status status--ok">
      {health.data.model_name} v{health.data.model_version}
      <button
        type="button"
        className="button button--tiny"
        onClick={() => reload.mutate()}
        disabled={reload.isPending}
        title="Load the newest registered version without restarting the API"
      >
        {reload.isPending ? '…' : 'reload'}
      </button>
    </span>
  );
}

export function Layout() {
  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <h1>ML Dashboard</h1>
        </div>
        <nav className="app__nav">
          <NavLink to="/" end>
            Predict
          </NavLink>
          <NavLink to="/runs">Runs</NavLink>
        </nav>
        <ModelStatus />
      </header>

      <main className="app__main">
        <Outlet />
      </main>

      <footer className="app__footer">
        <a href="/docs" target="_blank" rel="noreferrer">
          API documentation
        </a>
      </footer>
    </div>
  );
}
