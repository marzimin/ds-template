/**
 * Live inference: discover the contract, fill it in, get a prediction back.
 */
import { FeatureForm } from '../components/FeatureForm';
import { ProbabilityBars } from '../components/MetricGrid';
import { PageHeader } from '../components/Section';
import { ErrorState, Loading } from '../components/States';
import { formatPrediction } from '../lib/format';
import { usePredict, usePredictSchema } from '../api/hooks';

export function PredictPage() {
  const schema = usePredictSchema();
  const prediction = usePredict();

  if (schema.isPending) return <Loading label="Reading the model's feature contract…" />;
  if (schema.isError) return <ErrorState error={schema.error} />;

  return (
    <>
      <PageHeader
        title="Predict"
        meta={
          <>
            Model <strong>{schema.data.model_name}</strong> version{' '}
            <strong>{schema.data.model_version}</strong>
          </>
        }
      />

      <p className="prose">
        These fields come from the signature logged when the model was trained, not from
        anything hardcoded here. Train on different data and this form changes with it.
      </p>

      {/* Keying on the model version discards the form and mounts a fresh one
          whenever a different model is loaded, so stale inputs from a previous
          dataset can never linger. */}
      <FeatureForm
        key={`${schema.data.model_name}-${schema.data.model_version}`}
        features={schema.data.features}
        submitting={prediction.isPending}
        onSubmit={(values) => prediction.mutate(values)}
      />

      {prediction.isError && <ErrorState error={prediction.error} />}

      {prediction.isSuccess && (
        <div className="result" role="status">
          <h3>Prediction</h3>
          <p className="result__value" title={String(prediction.data.prediction)}>
            {formatPrediction(prediction.data.prediction)}
          </p>

          {/* Absent for regressors, which have no classes to score. */}
          {prediction.data.probabilities && (
            <ProbabilityBars probabilities={prediction.data.probabilities} />
          )}

          <p className="result__meta">
            Served by {prediction.data.model_name} v{prediction.data.model_version}
          </p>
        </div>
      )}
    </>
  );
}
