/**
 * The three states every data-driven view needs: loading, error, and empty.
 *
 * Having them as shared components is what stops each page inventing its own
 * spinner and its own way of showing a failure, which is the usual reason a
 * dashboard feels inconsistent.
 */
import { isNoModelError } from '../api/client';

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <p className="state state--loading" role="status">
      {label}
    </p>
  );
}

export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="state state--empty">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

/**
 * Render a failed request.
 *
 * A 503 is singled out deliberately. On a fresh checkout nobody has trained a
 * model yet, so the API returns 503 with instructions — that is an expected
 * first-run state, not a fault, and showing it in red as "Error" would tell the
 * user something is broken when nothing is.
 */
export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);

  if (isNoModelError(error)) {
    return (
      <div className="state state--guidance">
        <h3>No model has been trained yet</h3>
        <p>{message}</p>
        <p>
          Run the pipeline from a terminal, then reload this page:
          <code className="state__command">make pipeline</code>
        </p>
      </div>
    );
  }

  return (
    <div className="state state--error" role="alert">
      <h3>Something went wrong</h3>
      <p>{message}</p>
    </div>
  );
}
