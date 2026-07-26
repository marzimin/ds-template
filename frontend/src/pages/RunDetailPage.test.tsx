/**
 * Tests for the run detail page.
 *
 * Like the runs table, everything shown is derived from the response rather
 * than declared, so a pipeline logging different metrics or parameters still
 * displays without a code change.
 */
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { Route, Routes } from 'react-router';

import { RunDetailPage } from './RunDetailPage';
import { mockFetch, renderWithProviders } from '../test/utils';

afterEach(() => vi.restoreAllMocks());

/** Render at a URL so useParams() sees a run id, as the router would supply. */
function renderAt(runId: string) {
  return renderWithProviders(
    <Routes>
      <Route path="/runs/:runId" element={<RunDetailPage />} />
    </Routes>,
    { route: `/runs/${runId}` },
  );
}

const RUN = {
  run_id: 'abc123',
  run_name: 'nightly',
  status: 'FINISHED',
  start_time: 1735689600000,
  end_time: 1735689660000,
  metrics: { test_accuracy: 0.9561, train_accuracy: 1.0 },
  params: { model_name: 'xgboost', max_depth: '10' },
  tags: { 'mlflow.source.name': 'pipeline' },
};

it('shows metrics and parameters taken from the response', async () => {
  vi.stubGlobal(
    'fetch',
    mockFetch({ '/artifacts': { body: [] }, '/api/runs/abc123': { body: RUN } }),
  );
  renderAt('abc123');

  expect(await screen.findByText('nightly')).toBeInTheDocument();
  expect(screen.getByText('test_accuracy')).toBeInTheDocument();
  expect(screen.getByText('0.9561')).toBeInTheDocument();
  expect(screen.getByText('model_name')).toBeInTheDocument();
  expect(screen.getByText('xgboost')).toBeInTheDocument();
});

it('lists artifact folders as tabs and opens one on click', async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    'fetch',
    mockFetch({
      // Ordered so the more specific artifact route is matched first.
      '/artifacts?path=eda': { body: [{ path: 'eda/x.png', is_dir: false, file_size: 1 }] },
      '/artifacts': {
        body: [
          { path: 'eda', is_dir: true, file_size: null },
          { path: 'plots', is_dir: true, file_size: null },
        ],
      },
      '/api/runs/abc123': { body: RUN },
    }),
  );
  renderAt('abc123');

  const tab = await screen.findByRole('tab', { name: 'eda' });
  expect(screen.getByRole('tab', { name: 'plots' })).toBeInTheDocument();
  expect(tab).toHaveAttribute('aria-selected', 'false');

  await user.click(tab);
  expect(tab).toHaveAttribute('aria-selected', 'true');
});

it('reports a missing run as an error', async () => {
  vi.stubGlobal(
    'fetch',
    mockFetch({
      '/api/runs/missing': { status: 404, body: { detail: "Run 'missing' was not found." } },
    }),
  );
  renderAt('missing');

  expect(await screen.findByRole('alert')).toHaveTextContent('was not found');
});

it('handles a run with no metrics or parameters', async () => {
  vi.stubGlobal(
    'fetch',
    mockFetch({
      '/artifacts': { body: [] },
      '/api/runs/bare': { body: { ...RUN, run_id: 'bare', metrics: {}, params: {} } },
    }),
  );
  renderAt('bare');

  expect(await screen.findByText('No metrics were logged for this run.')).toBeInTheDocument();
  expect(screen.getByText('No parameters were logged.')).toBeInTheDocument();
});
