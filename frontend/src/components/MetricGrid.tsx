/**
 * Metric display: a grid of tiles, and a probability bar.
 *
 * Both route their numbers through `lib/format`, so a bounded score and an
 * error term in the target's units are each shown at a sensible precision
 * without either component knowing which metrics exist.
 */
import { formatMetric, formatPercent } from '../lib/format';

/**
 * A grid of metric tiles.
 *
 * Metrics arrive as a plain mapping from the API, so whatever the pipeline
 * logged is what appears — accuracy and f1 for a classifier, RMSE and R² for a
 * regressor, and anything added later with no change here.
 */
export function MetricGrid({ metrics }: { metrics: Record<string, number> }) {
  const entries = Object.entries(metrics).sort(([a], [b]) => a.localeCompare(b));

  return (
    <div className="metric-grid">
      {entries.map(([key, value]) => (
        <div key={key} className="metric">
          <span className="metric__label">{key}</span>
          <span className="metric__value" title={String(value)}>
            {formatMetric(value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * Per-class probabilities as labelled bars.
 *
 * Rendered only when the model produced them: a regressor has no classes, and
 * the API returns null rather than an empty object in that case.
 */
export function ProbabilityBars({ probabilities }: { probabilities: Record<string, number> }) {
  const entries = Object.entries(probabilities);

  return (
    <table className="table">
      <caption>Class probabilities</caption>
      <thead>
        <tr>
          <th scope="col">Class</th>
          <th scope="col">Probability</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([label, score]) => (
          <tr key={label}>
            <th scope="row">{label}</th>
            <td>
              <div
                className="bar"
                role="meter"
                aria-valuenow={Math.round(score * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`Probability of class ${label}`}
              >
                <div className="bar__fill" style={{ width: `${score * 100}%` }} />
                <span className="bar__label">{formatPercent(score)}</span>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
