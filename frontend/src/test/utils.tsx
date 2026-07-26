/** Test helpers: render a component with the providers the app supplies. */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import type { ReactElement } from 'react';
import { MemoryRouter } from 'react-router-dom';

export function renderWithProviders(ui: ReactElement, { route = '/' } = {}) {
  const queryClient = new QueryClient({
    // Retries would make a failing test wait for backoff before reporting.
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

/** Stub `fetch` with a handler keyed by URL substring. */
export function mockFetch(routes: Record<string, { status?: number; body: unknown }>) {
  return vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    const key = Object.keys(routes).find((candidate) => url.includes(candidate));
    const match = key ? routes[key] : undefined;

    if (!match) {
      return Promise.resolve(
        new Response(JSON.stringify({ detail: `no stub for ${url}` }), { status: 404 }),
      );
    }
    return Promise.resolve(
      new Response(JSON.stringify(match.body), {
        status: match.status ?? 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
}
