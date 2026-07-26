/**
 * Tests for the prediction page.
 *
 * The central claim being verified is that the form is built from whatever the
 * API reports, so a different dataset produces a different form with no code
 * change. Two different schemas are rendered to prove it.
 */
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PredictPage } from './PredictPage';
import { mockFetch, renderWithProviders } from '../test/utils';

const BREAST_CANCER_SCHEMA = {
  model_name: 'ds-template',
  model_version: '1',
  features: [
    {
      name: 'MEAN_RADIUS',
      mlflow_type: 'double',
      kind: 'number',
      required: true,
      example: 17.99,
    },
    {
      name: 'MEAN_TEXTURE',
      mlflow_type: 'double',
      kind: 'number',
      required: true,
      example: 10.38,
    },
  ],
};

const HOUSING_SCHEMA = {
  model_name: 'house-prices',
  model_version: '4',
  features: [
    {
      name: 'SUBURB',
      mlflow_type: 'string',
      kind: 'string',
      required: true,
      example: 'Camberwell',
    },
    { name: 'ROOMS', mlflow_type: 'long', kind: 'integer', required: true, example: 3 },
    {
      name: 'HAS_GARAGE',
      mlflow_type: 'boolean',
      kind: 'boolean',
      required: false,
      example: null,
    },
  ],
};

afterEach(() => vi.restoreAllMocks());

describe('form generated from the model signature', () => {
  it('renders one input per reported feature, pre-filled from the example', async () => {
    vi.stubGlobal(
      'fetch',
      mockFetch({ '/api/predict/schema': { body: BREAST_CANCER_SCHEMA } }),
    );
    renderWithProviders(<PredictPage />);

    expect(await screen.findByLabelText(/MEAN_RADIUS/)).toHaveValue(17.99);
    expect(screen.getByLabelText(/MEAN_TEXTURE/)).toHaveValue(10.38);
    expect(screen.getByText('2 features')).toBeInTheDocument();
  });

  it('renders a completely different form for a different dataset', async () => {
    vi.stubGlobal('fetch', mockFetch({ '/api/predict/schema': { body: HOUSING_SCHEMA } }));
    renderWithProviders(<PredictPage />);

    // A text input, a numeric input, and a boolean dropdown — chosen from the
    // declared kinds, with no knowledge of this dataset anywhere in the code.
    expect(await screen.findByLabelText(/SUBURB/)).toHaveValue('Camberwell');
    expect(screen.getByLabelText(/ROOMS/)).toHaveValue(3);
    expect(screen.getByLabelText(/HAS_GARAGE/).tagName).toBe('SELECT');
    expect(screen.queryByLabelText(/MEAN_RADIUS/)).not.toBeInTheDocument();
  });
});

describe('submitting a prediction', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      mockFetch({
        '/api/predict/schema': { body: BREAST_CANCER_SCHEMA },
        '/api/predict': {
          body: {
            prediction: 1,
            probabilities: { '0': 0.03, '1': 0.97 },
            model_name: 'ds-template',
            model_version: '1',
          },
        },
      }),
    );
  });

  it('shows the prediction and its class probabilities', async () => {
    const user = userEvent.setup();
    renderWithProviders(<PredictPage />);

    await user.click(await screen.findByRole('button', { name: 'Predict' }));

    expect(await screen.findByText('Prediction')).toBeInTheDocument();
    expect(screen.getByText('97.0%')).toBeInTheDocument();
    expect(screen.getByText('3.0%')).toBeInTheDocument();
  });

  it('sends values coerced to numbers, not strings', async () => {
    const user = userEvent.setup();
    renderWithProviders(<PredictPage />);

    await user.click(await screen.findByRole('button', { name: 'Predict' }));

    await waitFor(() => {
      const call = vi
        .mocked(fetch)
        .mock.calls.find(([url]) => String(url).endsWith('/api/predict'));
      expect(call).toBeDefined();
      const body = JSON.parse(String(call?.[1]?.body));
      expect(body.features.MEAN_RADIUS).toBe(17.99);
      expect(typeof body.features.MEAN_RADIUS).toBe('number');
    });
  });
});

describe('before a model exists', () => {
  it('explains how to train one instead of showing an error', async () => {
    vi.stubGlobal(
      'fetch',
      mockFetch({
        '/api/predict/schema': {
          status: 503,
          body: { detail: 'No versions registered. Train one first — run `make pipeline`.' },
        },
      }),
    );
    renderWithProviders(<PredictPage />);

    expect(await screen.findByText('No model has been trained yet')).toBeInTheDocument();
    expect(screen.getByText('make pipeline')).toBeInTheDocument();
    // A first-run state must not be presented as a failure.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows a real error as an error', async () => {
    vi.stubGlobal(
      'fetch',
      mockFetch({
        '/api/predict/schema': { status: 500, body: { detail: 'boom' } },
      }),
    );
    renderWithProviders(<PredictPage />);

    // A 500 may be transient, so the app retries it twice with backoff before
    // giving up — unlike a 503, which it reports immediately. Allow for that.
    expect(await screen.findByRole('alert', {}, { timeout: 8000 })).toHaveTextContent('boom');
  }, 10000);
});
