/**
 * Displays the image artifacts inside one folder of a run.
 *
 * Images are rendered from their API URL directly — the browser fetches each
 * one as an ordinary `<img src>`, so the backend's artifact endpoint doubles as
 * an image host. Non-image files are offered as links instead of being guessed
 * at.
 */
import { artifactFileUrl } from '../api/client';
import { EmptyState, ErrorState, Loading } from './States';
import { useArtifacts } from '../api/hooks';

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp'];

function isImage(path: string): boolean {
  return IMAGE_EXTENSIONS.some((extension) => path.toLowerCase().endsWith(extension));
}

export function ArtifactGallery({ runId, path }: { runId: string; path: string }) {
  const artifacts = useArtifacts(runId, path);

  if (artifacts.isPending) return <Loading label={`Loading ${path}…`} />;
  if (artifacts.isError) return <ErrorState error={artifacts.error} />;

  const files = artifacts.data.filter((entry) => !entry.is_dir);
  if (files.length === 0) {
    return (
      <EmptyState title={`Nothing in ${path}`}>
        <p>This folder contains no files.</p>
      </EmptyState>
    );
  }

  const images = files.filter((entry) => isImage(entry.path));
  const others = files.filter((entry) => !isImage(entry.path));

  return (
    <div className="gallery-wrapper">
      <p className="page-header__meta">
        {files.length} file{files.length === 1 ? '' : 's'} in <code>{path}</code>
      </p>

      {images.length > 0 && (
        <div className="gallery">
          {images.map((entry) => (
            <figure key={entry.path} className="gallery__item">
              <a href={artifactFileUrl(runId, entry.path)} target="_blank" rel="noreferrer">
                <img
                  src={artifactFileUrl(runId, entry.path)}
                  alt={entry.path}
                  loading="lazy"
                  className="gallery__image"
                />
              </a>
              <figcaption className="gallery__caption">
                {entry.path.split('/').pop()}
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      {others.length > 0 && (
        <ul className="file-list">
          {others.map((entry) => (
            <li key={entry.path}>
              <a href={artifactFileUrl(runId, entry.path)} target="_blank" rel="noreferrer">
                {entry.path.split('/').pop()}
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
