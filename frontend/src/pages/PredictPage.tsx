/**
 * Live inference: discover the contract, fill it in, get a prediction back.
 */
import { FeatureForm } from '../components/FeatureForm';
import { ErrorState, Loading } from '../components/States';
import { usePredict, usePredictSchema } from '../api/hooks';

export function PredictPage() {
  const schema = usePredictSchema();
  const prediction = usePredict();

  if (schema.isPending) return <Loading label="Reading the model's feature contract…" />;
  if (schema.isError) return <ErrorState error={schema.error} />;

  return (
    <section>
      <header className="page-header">
        <h2>Predict</h2>
        <p className="page-header__meta">
          Model <strong>{schema.data.model_name}</strong> version{' '}
          <strong>{schema.data.model_version}</strong>
        </p>
      </header>

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
          <p className="result__value">{String(prediction.data.prediction)}</p>

          {prediction.data.probabilities && (
            <table className="table">
              <caption>Class probabilities</caption>
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col">Probability</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(prediction.data.probabilities).map(([label, score]) => (
                  <tr key={label}>
                    <th scope="row">{label}</th>
                    <td>
                      <div className="bar">
                        <div className="bar__fill" style={{ width: `${score * 100}%` }} />
                        <span className="bar__label">{(score * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <p className="result__meta">
            Served by {prediction.data.model_name} v{prediction.data.model_version}
          </p>
        </div>
      )}
    </section>
  );
}
