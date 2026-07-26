/**
 * One run: its metrics, its parameters, and the plots it produced.
 *
 * Artifacts are listed from MLflow rather than read from a local directory, so
 * this keeps working when the API runs somewhere else entirely.
 */
import { useState } from 'react';
import { Link, useParams } from 'react-router';

import { ArtifactGallery } from '../components/ArtifactGallery';
import { EmptyState, ErrorState, Loading } from '../components/States';
import { useArtifacts, useRun } from '../api/hooks';

export function RunDetailPage() {
  const { runId = '' } = useParams();
  const run = useRun(runId);
  const topLevel = useArtifacts(runId);
  const [folder, setFolder] = useState<string | null>(null);

  if (run.isPending) return <Loading label="Loading run…" />;
  if (run.isError) return <ErrorState error={run.error} />;

  const metrics = Object.entries(run.data.metrics ?? {}).sort(([a], [b]) => a.localeCompare(b));
  const params = Object.entries(run.data.params ?? {}).sort(([a], [b]) => a.localeCompare(b));
  const folders = (topLevel.data ?? []).filter((entry) => entry.is_dir);

  return (
    <section>
      <header className="page-header">
        <div>
          <h2>{run.data.run_name ?? run.data.run_id.slice(0, 8)}</h2>
          <p className="page-header__meta">
            <code>{run.data.run_id}</code> · {run.data.status ?? 'unknown'}
          </p>
        </div>
        <Link to="/runs" className="button button--ghost">
          ← All runs
        </Link>
      </header>

      <h3>Metrics</h3>
      {metrics.length === 0 ? (
        <p className="prose">No metrics were logged for this run.</p>
      ) : (
        <div className="metric-grid">
          {metrics.map(([key, value]) => (
            <div key={key} className="metric">
              <span className="metric__label">{key}</span>
              <span className="metric__value">{value.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}

      <h3>Parameters</h3>
      {params.length === 0 ? (
        <p className="prose">No parameters were logged.</p>
      ) : (
        <div className="table-scroll">
          <table className="table">
            <tbody>
              {params.map(([key, value]) => (
                <tr key={key}>
                  <th scope="row">{key}</th>
                  <td>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h3>Artifacts</h3>
      {topLevel.isPending && <Loading label="Loading artifacts…" />}
      {topLevel.isError && <ErrorState error={topLevel.error} />}
      {topLevel.isSuccess && folders.length === 0 && (
        <EmptyState title="No artifacts">
          <p>This run logged no files.</p>
        </EmptyState>
      )}

      {folders.length > 0 && (
        <>
          <div className="tabs" role="tablist">
            {folders.map((entry) => (
              <button
                key={entry.path}
                role="tab"
                aria-selected={folder === entry.path}
                className={`tab ${folder === entry.path ? 'tab--active' : ''}`}
                onClick={() => setFolder(folder === entry.path ? null : entry.path)}
              >
                {entry.path}
              </button>
            ))}
          </div>
          {folder && <ArtifactGallery runId={runId} path={folder} />}
        </>
      )}
    </section>
  );
}
