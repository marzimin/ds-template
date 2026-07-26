/**
 * Training run history.
 *
 * Metric columns are derived from whatever the runs actually logged rather than
 * being a fixed list, so a pipeline that adds a metric shows it immediately and
 * one that renames a metric does not leave an empty column behind.
 */
import { Link } from 'react-router-dom';

import { EmptyState, ErrorState, Loading } from '../components/States';
import { useRuns } from '../api/hooks';
import type { RunSummary } from '../api/client';

function metricColumns(runs: RunSummary[]): string[] {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const key of Object.keys(run.metrics ?? {})) keys.add(key);
  }
  // Test metrics first: they are what you compare runs on.
  return [...keys].sort((a, b) => {
    const aTest = a.startsWith('test_') ? 0 : 1;
    const bTest = b.startsWith('test_') ? 0 : 1;
    return aTest - bTest || a.localeCompare(b);
  });
}

function formatTime(epochMs: number | null | undefined): string {
  if (!epochMs) return '—';
  return new Date(epochMs).toLocaleString();
}

export function RunsPage() {
  const runs = useRuns();

  if (runs.isPending) return <Loading label="Loading runs…" />;
  if (runs.isError) return <ErrorState error={runs.error} />;

  if (runs.data.length === 0) {
    return (
      <EmptyState title="No runs yet">
        <p>
          Nothing has been logged to MLflow for this experiment. Run the pipeline from a
          terminal:
        </p>
        <code className="state__command">make pipeline</code>
      </EmptyState>
    );
  }

  const columns = metricColumns(runs.data);

  return (
    <section>
      <header className="page-header">
        <h2>Runs</h2>
        <p className="page-header__meta">{runs.data.length} run(s), newest first</p>
      </header>

      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Run</th>
              <th scope="col">Status</th>
              <th scope="col">Started</th>
              {columns.map((key) => (
                <th scope="col" key={key}>
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.data.map((run) => (
              <tr key={run.run_id}>
                <th scope="row">
                  <Link to={`/runs/${run.run_id}`}>
                    {run.run_name ?? run.run_id.slice(0, 8)}
                  </Link>
                </th>
                <td>{run.status ?? '—'}</td>
                <td>{formatTime(run.start_time)}</td>
                {columns.map((key) => {
                  const value = run.metrics?.[key];
                  return (
                    <td key={key} className="table__number">
                      {value == null ? '—' : value.toFixed(4)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
