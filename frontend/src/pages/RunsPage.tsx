/**
 * Training run history.
 *
 * Metric columns are derived from what the runs actually logged rather than
 * declared here, so a pipeline that adds a metric shows it immediately and a
 * regression pipeline logging RMSE needs no change. Values are formatted by
 * magnitude, which is what lets a bounded score and an error term in the
 * target's units share one table.
 */
import { Link } from 'react-router';

import { DataTable, type Column } from '../components/DataTable';
import { PageHeader } from '../components/Section';
import { EmptyState, ErrorState, Loading } from '../components/States';
import { formatMetric, formatTimestamp, formatTimestampExact } from '../lib/format';
import { useRuns } from '../api/hooks';
import type { RunSummary } from '../api/client';

/** Union of every metric key across the runs, test metrics first. */
function metricColumns(runs: RunSummary[]): string[] {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const key of Object.keys(run.metrics ?? {})) keys.add(key);
  }
  return [...keys].sort((a, b) => {
    const aTest = a.startsWith('test_') ? 0 : 1;
    const bTest = b.startsWith('test_') ? 0 : 1;
    return aTest - bTest || a.localeCompare(b);
  });
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

  const columns: Column<RunSummary>[] = [
    {
      key: 'run',
      header: 'Run',
      rowHeader: true,
      render: (run) => (
        <Link to={`/runs/${run.run_id}`}>{run.run_name ?? run.run_id.slice(0, 8)}</Link>
      ),
    },
    { key: 'status', header: 'Status', render: (run) => run.status ?? '—' },
    {
      key: 'started',
      header: 'Started',
      render: (run) => (
        <time title={formatTimestampExact(run.start_time)}>
          {formatTimestamp(run.start_time)}
        </time>
      ),
    },
    ...metricColumns(runs.data).map((key): Column<RunSummary> => ({
      key,
      header: key,
      numeric: true,
      render: (run) => {
        const value = run.metrics?.[key];
        return value == null ? '—' : formatMetric(value);
      },
    })),
  ];

  return (
    <>
      <PageHeader title="Runs" meta={`${runs.data.length} run(s), newest first`} />
      <DataTable columns={columns} rows={runs.data} rowKey={(run) => run.run_id} />
    </>
  );
}
