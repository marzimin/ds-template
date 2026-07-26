/**
 * A prediction form built at runtime from the model's signature.
 *
 * This component never mentions a feature name. It receives the list the API
 * reported from `GET /api/predict/schema` and renders one input per entry,
 * choosing the control from the declared `kind`. Swap the dataset, retrain, and
 * the form redraws itself with the new columns — no code change here.
 *
 * That indirection is the single most important design decision in this
 * frontend. Hardcoding thirty numeric inputs for the demo dataset would make
 * the template useless for anyone else's data.
 */
import { useState } from 'react';

import type { FeatureSpec } from '../api/client';

interface FeatureFormProps {
  features: FeatureSpec[];
  onSubmit: (values: Record<string, unknown>) => void;
  submitting?: boolean;
}

/** Form state is kept as strings because that is what DOM inputs produce. */
type FormValues = Record<string, string>;

function initialValues(features: FeatureSpec[]): FormValues {
  const values: FormValues = {};
  for (const feature of features) {
    values[feature.name] = feature.example == null ? '' : String(feature.example);
  }
  return values;
}

/**
 * Convert a form string back to the type the model expects.
 *
 * The backend validates properly and returns a 422 with a clear message, so
 * this only has to get the common cases right rather than duplicate the
 * server's rules — a rule kept deliberately: validation lives in one place.
 */
function coerce(feature: FeatureSpec, raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === '') return null;

  switch (feature.kind) {
    case 'number':
    case 'integer': {
      const parsed = Number(trimmed);
      return Number.isNaN(parsed) ? trimmed : parsed;
    }
    case 'boolean':
      return trimmed.toLowerCase() === 'true';
    default:
      return trimmed;
  }
}

function inputType(kind: string): string {
  if (kind === 'number' || kind === 'integer') return 'number';
  if (kind === 'datetime') return 'datetime-local';
  return 'text';
}

/**
 * Retraining can change the feature list while the page is open, and the form
 * must not keep showing inputs the current model no longer accepts. The caller
 * handles that by passing a `key` derived from the model version, which makes
 * React discard this component and mount a fresh one — the documented way to
 * reset all state when a prop changes, and simpler than syncing in an effect.
 */
export function FeatureForm({ features, onSubmit, submitting = false }: FeatureFormProps) {
  const [values, setValues] = useState<FormValues>(() => initialValues(features));

  const hasExamples = features.some((feature) => feature.example != null);

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const payload: Record<string, unknown> = {};
    for (const feature of features) {
      payload[feature.name] = coerce(feature, values[feature.name] ?? '');
    }
    onSubmit(payload);
  }

  return (
    <form onSubmit={handleSubmit} className="feature-form">
      <div className="feature-form__toolbar">
        <p className="feature-form__count">
          {features.length} feature{features.length === 1 ? '' : 's'}
        </p>
        <div className="feature-form__actions">
          {hasExamples && (
            <button
              type="button"
              className="button button--ghost"
              onClick={() => setValues(initialValues(features))}
            >
              Reset to example
            </button>
          )}
          <button
            type="button"
            className="button button--ghost"
            onClick={() => setValues(Object.fromEntries(features.map((f) => [f.name, ''])))}
          >
            Clear
          </button>
          <button type="submit" className="button" disabled={submitting}>
            {submitting ? 'Predicting…' : 'Predict'}
          </button>
        </div>
      </div>

      <div className="feature-form__grid">
        {features.map((feature) => {
          const id = `feature-${feature.name}`;
          return (
            <div key={feature.name} className="field">
              <label htmlFor={id} className="field__label">
                {feature.name}
                {feature.required && (
                  <span className="field__required" aria-label="required">
                    *
                  </span>
                )}
              </label>
              {feature.kind === 'boolean' ? (
                <select
                  id={id}
                  className="field__input"
                  value={values[feature.name] ?? ''}
                  onChange={(event) =>
                    setValues((prev) => ({ ...prev, [feature.name]: event.target.value }))
                  }
                >
                  <option value="">—</option>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              ) : (
                <input
                  id={id}
                  className="field__input"
                  type={inputType(feature.kind)}
                  inputMode={feature.kind === 'integer' ? 'numeric' : undefined}
                  step={feature.kind === 'number' ? 'any' : undefined}
                  value={values[feature.name] ?? ''}
                  onChange={(event) =>
                    setValues((prev) => ({ ...prev, [feature.name]: event.target.value }))
                  }
                />
              )}
              <span className="field__hint">{feature.mlflow_type}</span>
            </div>
          );
        })}
      </div>
    </form>
  );
}
