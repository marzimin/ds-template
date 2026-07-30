/**
 * One run: its metrics, its parameters, and the plots it produced.
 *
 * Artifacts are listed from MLflow rather than read from a local directory, so
 * this keeps working when the API runs somewhere else entirely.
 */
import { useState } from 'react';
import { Link, useParams } from 'react-router';

import { ArtifactGallery } from '../components/ArtifactGallery';
import { KeyValueTable } from '../components/DataTable';
import { MetricGrid } from '../components/MetricGrid';
import { PageHeader, Section } from '../components/Section';
import { ErrorState, Loading } from '../components/States';
import { useArtifacts, useRun } from '../api/hooks';

export function RunDetailPage() {
  const { runId = '' } = useParams();
  const run = useRun(runId);
  const topLevel = useArtifacts(runId);
  const [folder, setFolder] = useState<string | null>(null);

  if (run.isPending) return <Loading label="Loading run…" />;
  if (run.isError) return <ErrorState error={run.error} />;

  const metrics = run.data.metrics ?? {};
  const params = Object.entries(run.data.params ?? {}).sort(([a], [b]) => a.localeCompare(b));
  const folders = (topLevel.data ?? []).filter((entry) => entry.is_dir);

  return (
    <>
      <PageHeader
        title={run.data.run_name ?? run.data.run_id.slice(0, 8)}
        meta={
          <>
            <code>{run.data.run_id}</code> · {run.data.status ?? 'unknown'}
          </>
        }
        actions={
          <Link to="/runs" className="button button--ghost">
            ← All runs
          </Link>
        }
      />

      <Section
        title="Metrics"
        empty={Object.keys(metrics).length === 0 && 'No metrics were logged'}
      >
        <MetricGrid metrics={metrics} />
      </Section>

      <Section title="Parameters" empty={params.length === 0 && 'No parameters were logged'}>
        <KeyValueTable entries={params} />
      </Section>

      <Section
        title="Artifacts"
        empty={topLevel.isSuccess && folders.length === 0 && 'This run logged no files'}
      >
        {topLevel.isPending && <Loading label="Loading artifacts…" />}
        {topLevel.isError && <ErrorState error={topLevel.error} />}
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
      </Section>
    </>
  );
}
