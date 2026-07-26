/**
 * Tests for the artifact gallery.
 *
 * Two behaviours matter here: images render inline while other files become
 * links, and every artifact URL is built through `artifactFileUrl` so the
 * run id and path are encoded rather than interpolated raw.
 */
import { screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { ArtifactGallery } from './ArtifactGallery';
import { mockFetch, renderWithProviders } from '../test/utils';

afterEach(() => vi.restoreAllMocks());

const entry = (path: string, is_dir = false) => ({ path, is_dir, file_size: 100 });

it('renders images inline and other files as links', async () => {
  vi.stubGlobal(
    'fetch',
    mockFetch({
      '/artifacts': {
        body: [entry('eda/hist.png'), entry('reports/report.txt'), entry('nested', true)],
      },
    }),
  );
  renderWithProviders(<ArtifactGallery runId="run-1" path="eda" />);

  const image = await screen.findByAltText('eda/hist.png');
  expect(image).toBeInstanceOf(HTMLImageElement);
  expect(image).toHaveAttribute('loading', 'lazy');

  // Non-images are offered as links rather than rendered as broken images.
  expect(screen.getByRole('link', { name: 'report.txt' })).toBeInTheDocument();
  expect(screen.queryByAltText('reports/report.txt')).not.toBeInTheDocument();

  // Directories are not files and must not appear as either.
  expect(screen.queryByText('nested')).not.toBeInTheDocument();
});

it('percent-encodes the artifact path in the URL', async () => {
  vi.stubGlobal('fetch', mockFetch({ '/artifacts': { body: [entry('eda/a b&c.png')] } }));
  renderWithProviders(<ArtifactGallery runId="run 1" path="eda" />);

  const image = await screen.findByAltText('eda/a b&c.png');
  const src = image.getAttribute('src') ?? '';
  expect(src).toContain('run%201');
  expect(src).toContain('eda%2Fa%20b%26c.png');
});

it('shows an empty state when a folder has no files', async () => {
  vi.stubGlobal('fetch', mockFetch({ '/artifacts': { body: [entry('sub', true)] } }));
  renderWithProviders(<ArtifactGallery runId="run-1" path="eda" />);

  expect(await screen.findByText('Nothing in eda')).toBeInTheDocument();
});

it('surfaces a failure rather than rendering an empty gallery', async () => {
  vi.stubGlobal(
    'fetch',
    mockFetch({ '/artifacts': { status: 404, body: { detail: 'Run not found.' } } }),
  );
  renderWithProviders(<ArtifactGallery runId="nope" path="eda" />);

  expect(await screen.findByRole('alert')).toHaveTextContent('Run not found.');
});
