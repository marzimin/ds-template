/** Routing table: which URL shows which page. */
import { Navigate, Route, Routes } from 'react-router';

import { Layout } from './components/Layout';
import { PredictPage } from './pages/PredictPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { RunsPage } from './pages/RunsPage';

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<PredictPage />} />
        <Route path="runs" element={<RunsPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
