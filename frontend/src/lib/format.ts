/**
 * How this application turns numbers into text — decided once, here.
 *
 * The template supports classification and regression, so two very different
 * kinds of number land in the same table. A bounded score sits in [0, 1] and
 * wants fixed decimals; an error term is in the units of the target and might
 * be 54.75 or 128,456.79. A single `toFixed(4)` cannot serve both: it adds
 * noise to large values and rounds small ones away entirely, so a metric of
 * 0.00003 renders as "0.0000".
 *
 * Formatting is chosen from the **magnitude of the value**, never from the
 * metric's name. Matching on names like "accuracy" or "rmse" would tie the UI
 * to one project's metric vocabulary, and the point of this template is that a
 * pipeline logging `test_mape` or `test_spearman` displays without any
 * frontend change.
 */

/** Below this, fixed notation would round a value away to "0.0000". */
const TINY = 1e-3;

/**
 * Format a metric for display.
 *
 * @param value - The metric value.
 * @param locale - Overrides the browser locale; used by tests for stability.
 * @returns A string sized to the value's magnitude.
 *
 * @example
 * formatMetric(0.9561)      // "0.9561"   a score keeps four decimals
 * formatMetric(54.752601)   // "54.75"    an error term keeps two
 * formatMetric(128456.789)  // "128,456.79"
 * formatMetric(0.00003)     // "3.00e-5"  never silently zero
 */
export function formatMetric(value: number, locale?: string): string {
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';

  const magnitude = Math.abs(value);

  // Too small for fixed notation: exponential keeps the information rather
  // than rounding it to zero.
  if (magnitude < TINY) return value.toExponential(2);

  // Scores and probabilities live here. Four decimals is the convention for
  // reporting them and enough to separate two close models.
  if (magnitude <= 1) return value.toFixed(4);

  // Everything else is in the units of the target, where two decimals is
  // plenty and thousands separators do the real readability work.
  return value.toLocaleString(locale, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/**
 * Format a model's prediction.
 *
 * A classifier returns a label — an integer, a string, a boolean — which should
 * appear exactly as given. A regressor returns a float whose full binary
 * precision is an implementation detail: `String(152.13381958007812)` prints
 * seventeen digits, which reads as false confidence.
 *
 * @param value - Whatever the API returned as the prediction.
 * @param locale - Overrides the browser locale; used by tests for stability.
 * @returns A display string.
 */
export function formatPrediction(value: unknown, locale?: string): string {
  if (value == null) return '—';
  if (typeof value === 'number') {
    // Whole numbers are class labels stored as numbers; leave them alone.
    return Number.isInteger(value) ? String(value) : formatMetric(value, locale);
  }
  return String(value);
}

/**
 * Format a probability as a percentage.
 *
 * @param value - A probability in [0, 1].
 * @returns A percentage string to one decimal place.
 */
export function formatPercent(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format an MLflow timestamp.
 *
 * MLflow reports Unix epoch milliseconds. Recent runs are shown relatively,
 * because "3 hours ago" answers "is this the run I just started?" faster than
 * a full date does; older ones fall back to an absolute date.
 *
 * @param epochMs - Unix epoch milliseconds, or null.
 * @param now - Overrides the current time; used by tests for stability.
 * @returns A display string.
 */
export function formatTimestamp(
  epochMs: number | null | undefined,
  now: number = Date.now(),
): string {
  if (!epochMs) return '—';

  const seconds = Math.round((now - epochMs) / 1000);
  if (seconds < 0) return new Date(epochMs).toLocaleString();

  const units: [limit: number, size: number, name: Intl.RelativeTimeFormatUnit][] = [
    [60, 1, 'second'],
    [3600, 60, 'minute'],
    [86400, 3600, 'hour'],
    [604800, 86400, 'day'],
  ];

  for (const [limit, size, unit] of units) {
    if (seconds < limit) {
      const relative = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
      return relative.format(-Math.floor(seconds / size), unit);
    }
  }

  return new Date(epochMs).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/** Absolute timestamp, for the `title` attribute behind a relative one. */
export function formatTimestampExact(epochMs: number | null | undefined): string {
  return epochMs ? new Date(epochMs).toLocaleString() : '';
}
