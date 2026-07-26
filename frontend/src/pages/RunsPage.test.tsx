/**
 * Tests for the runs dashboard.
 *
 * The claim being verified is that metric columns come from the data rather
 * than a fixed list, so a pipeline logging different metrics still displays.
 */
import { screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { RunsPage } from './RunsPage';
import { mockFetch, renderWithProviders } from '../test/utils';

afterEach(() => vi.restoreAllMocks());

const run = (overrides: Record<string, unknown> = {}) => ({
  run_id: 'abc123def456',
  run_name: 'nightly',
  status: 'FINISHED',
  start_time: 1735689600000,
  end_time: 1735689660000,
  metrics: { test_accuracy: 0.9561, train_accuracy: 1.0 },
  ...overrides,
});

it('derives metric columns from the runs themselves', async () => {
  vi.stubGlobal('fetch', mockFetch({ '/api/runs': { body: [run()] } }));
  renderWithProviders(<RunsPage />);

  expect(await screen.findByText('test_accuracy')).toBeInTheDocument();
  expect(screen.getByText('train_accuracy')).toBeInTheDocument();
  expect(screen.getByText('0.9561')).toBeInTheDocument();
});

it('shows metrics this template never ships with', async () => {
  // A regression pipeline logging RMSE must display without a code change.
  vi.stubGlobal(
    'fetch',
    mockFetch({
      '/api/runs': { body: [run({ metrics: { test_rmse: 12.5, test_r2: 0.83 } })] },
    }),
  );
  renderWithProviders(<RunsPage />);

  expect(await screen.findByText('test_rmse')).toBeInTheDocument();
  expect(screen.getByText('test_r2')).toBeInTheDocument();
});

it('fills gaps when runs logged different metrics', async () => {
  vi.stubGlobal(
    'fetch',
    mockFetch({
      '/api/runs': {
        body: [
          run({ run_id: 'r1', metrics: { test_accuracy: 0.9 } }),
          run({ run_id: 'r2', run_name: 'older', metrics: { test_f1_score: 0.8 } }),
        ],
      },
    }),
  );
  renderWithProviders(<RunsPage />);

  // Both columns exist; the missing cell shows a dash rather than breaking.
  expect(await screen.findByText('test_accuracy')).toBeInTheDocument();
  expect(screen.getByText('test_f1_score')).toBeInTheDocument();
  expect(screen.getAllByText('—').length).toBeGreaterThan(0);
});

it('shows an empty state, not an error, when nothing has run', async () => {
  vi.stubGlobal('fetch', mockFetch({ '/api/runs': { body: [] } }));
  renderWithProviders(<RunsPage />);

  expect(await screen.findByText('No runs yet')).toBeInTheDocument();
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});
