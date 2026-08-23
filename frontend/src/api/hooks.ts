/**
 * TanStack Query hooks — one per endpoint.
 *
 * These handle loading, error, caching, and refetching so components never
 * manage that state themselves. A component asks for data and renders one of
 * three cases: loading, error, or loaded.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { api, type PredictResponse } from './client';

export const queryKeys = {
  health: ['health'] as const,
  predictSchema: ['predict', 'schema'] as const,
  runs: (limit: number) => ['runs', limit] as const,
  run: (runId: string) => ['runs', runId] as const,
  artifacts: (runId: string, path: string) => ['runs', runId, 'artifacts', path] as const,
};

/**
 * Do not retry a 503. It means no model has been trained, which will not
 * resolve by asking again, and retrying only delays showing the explanation.
 */
function retryUnlessExpected(failureCount: number, error: unknown): boolean {
  const status = (error as { status?: number })?.status;
  if (status === 503 || status === 404 || status === 422) return false;
  return failureCount < 2;
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.health,
    refetchInterval: 30_000,
  });
}

export function usePredictSchema() {
  return useQuery({
    queryKey: queryKeys.predictSchema,
    queryFn: api.predictSchema,
    retry: retryUnlessExpected,
  });
}

export function usePredict() {
  return useMutation<PredictResponse, Error, Record<string, unknown>>({
    mutationFn: api.predict,
  });
}

export function useReloadModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: api.reloadModel,
    // A new model version changes the feature contract and the health summary,
    // so drop both rather than leaving stale data on screen.
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.predictSchema });
      void queryClient.invalidateQueries({ queryKey: queryKeys.health });
    },
  });
}

export function useRuns(limit = 50) {
  return useQuery({
    queryKey: queryKeys.runs(limit),
    queryFn: () => api.runs(limit),
    retry: retryUnlessExpected,
  });
}

export function useDeleteRuns() {
  const queryClient = useQueryClient();
  return useMutation({
    // Runs are deleted one request per id rather than a bulk endpoint. Every
    // request is attempted even if one fails, so a partial failure still
    // deletes what it could; the failures are then reported together.
    mutationFn: async (runIds: string[]) => {
      const results = await Promise.allSettled(runIds.map((runId) => api.deleteRun(runId)));
      const failures = results.filter(
        (result): result is PromiseRejectedResult => result.status === 'rejected',
      );
      if (failures.length > 0) {
        const detail = failures[0]?.reason instanceof Error ? failures[0].reason.message : '';
        throw new Error(
          `Failed to delete ${failures.length} of ${runIds.length} run(s).${detail ? ` ${detail}` : ''}`,
        );
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
    },
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: queryKeys.run(runId),
    queryFn: () => api.run(runId),
    enabled: Boolean(runId),
    retry: retryUnlessExpected,
  });
}

export function useArtifacts(runId: string, path = '') {
  return useQuery({
    queryKey: queryKeys.artifacts(runId, path),
    queryFn: () => api.artifacts(runId, path),
    enabled: Boolean(runId),
    retry: retryUnlessExpected,
  });
}
