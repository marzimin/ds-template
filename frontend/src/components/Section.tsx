/**
 * Layout primitives shared by every page.
 *
 * These exist so the three screens are structurally identical rather than
 * separately written. Adding a page means composing these, not reproducing the
 * markup — which is what keeps a template feeling like one system as it grows.
 */
import type { ReactNode } from 'react';

import { EmptyState } from './States';

interface PageHeaderProps {
  title: string;
  /** Secondary line under the title: model version, run id, counts. */
  meta?: ReactNode;
  /** Right-aligned controls, such as a back link. */
  actions?: ReactNode;
}

/** The title block at the top of a page. */
export function PageHeader({ title, meta, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div>
        <h2>{title}</h2>
        {meta && <p className="page-header__meta">{meta}</p>}
      </div>
      {actions}
    </header>
  );
}

interface SectionProps {
  title: string;
  /**
   * Shown in place of the children when there is nothing to display. Passing a
   * string gets the standard empty-state card, so every page reports "nothing
   * here" the same way instead of some using a card and others a bare
   * paragraph.
   */
  empty?: string | false;
  children: ReactNode;
}

/** A titled block of a page, with a consistent empty state. */
export function Section({ title, empty, children }: SectionProps) {
  return (
    <section className="section">
      <h3>{title}</h3>
      {empty ? <EmptyState title={empty} /> : children}
    </section>
  );
}
