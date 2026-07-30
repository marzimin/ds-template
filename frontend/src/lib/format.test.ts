/**
 * Tests for number formatting.
 *
 * These pin the behaviour that made the module necessary: one fixed precision
 * cannot serve both a bounded score and an error term in the target's units,
 * and the failure mode was silent — a small metric rendering as zero.
 */
import { describe, expect, it } from 'vitest';

import { formatMetric, formatPercent, formatPrediction, formatTimestamp } from './format';

describe('formatMetric', () => {
  it.each([
    [0.9561, '0.9561'],
    [0.4342, '0.4342'],
    [1, '1.0000'],
    [0.5, '0.5000'],
  ])('keeps four decimals for the bounded score %f', (value, expected) => {
    expect(formatMetric(value, 'en-GB')).toBe(expected);
  });

  it.each([
    [54.752601, '54.75'],
    [44.601918, '44.60'],
  ])('trims an error term in target units: %f', (value, expected) => {
    expect(formatMetric(value, 'en-GB')).toBe(expected);
  });

  it('groups digits in a large error term', () => {
    // "128456.7891" is a wall of digits; grouping is what makes it readable.
    expect(formatMetric(128456.7891, 'en-GB')).toBe('128,456.79');
  });

  it.each([[0.00003], [0.000091], [1.2e-7]])(
    'never rounds the small value %f away to zero',
    (value) => {
      const formatted = formatMetric(value, 'en-GB');
      expect(formatted).not.toBe('0.0000');
      expect(Number(formatted)).toBeCloseTo(value, 10);
    },
  );

  it('handles zero and non-finite values without producing noise', () => {
    expect(formatMetric(0)).toBe('0');
    expect(formatMetric(Number.NaN)).toBe('—');
    expect(formatMetric(Number.POSITIVE_INFINITY)).toBe('—');
  });

  it('formats negative values, which R² can be', () => {
    // A model worse than predicting the mean has R² below zero.
    expect(formatMetric(-0.3421, 'en-GB')).toBe('-0.3421');
    expect(formatMetric(-1234.5, 'en-GB')).toBe('-1,234.50');
  });
});

describe('formatPrediction', () => {
  it('leaves a class label exactly as it is', () => {
    expect(formatPrediction(1)).toBe('1');
    expect(formatPrediction(0)).toBe('0');
    expect(formatPrediction('setosa')).toBe('setosa');
  });

  it('trims the float precision of a regression prediction', () => {
    // A float32 model output widened to float64 prints its full binary
    // expansion through String(), which reads as false confidence.
    const raw = Math.fround(152.1338);
    expect(String(raw).length).toBeGreaterThan(8);
    expect(formatPrediction(raw, 'en-GB')).toBe('152.13');
  });

  it('renders a missing prediction as a dash', () => {
    expect(formatPrediction(null)).toBe('—');
    expect(formatPrediction(undefined)).toBe('—');
  });
});

describe('formatPercent', () => {
  it('renders a probability to one decimal place', () => {
    expect(formatPercent(0.9743)).toBe('97.4%');
    expect(formatPercent(0.0257)).toBe('2.6%');
    expect(formatPercent(0)).toBe('0.0%');
  });
});

describe('formatTimestamp', () => {
  const now = Date.UTC(2026, 0, 15, 12, 0, 0);

  it('shows recent runs relatively', () => {
    expect(formatTimestamp(now - 30_000, now)).toMatch(/second/);
    expect(formatTimestamp(now - 5 * 60_000, now)).toMatch(/minute/);
    expect(formatTimestamp(now - 3 * 3_600_000, now)).toMatch(/hour/);
  });

  it('falls back to an absolute date for older runs', () => {
    const monthsAgo = now - 90 * 86_400_000;
    expect(formatTimestamp(monthsAgo, now)).toMatch(/2025/);
  });

  it('renders a missing timestamp as a dash', () => {
    expect(formatTimestamp(null)).toBe('—');
    expect(formatTimestamp(undefined)).toBe('—');
  });
});
