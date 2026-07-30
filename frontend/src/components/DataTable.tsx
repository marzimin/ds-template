/**
 * Tabular display, shared so every table scrolls, aligns, and labels the same.
 *
 * Columns are described as data rather than written as markup. That is what
 * lets the runs dashboard build its columns from whatever metrics MLflow
 * reported, instead of declaring a fixed set that a regression pipeline would
 * leave empty.
 */
import type { ReactNode } from 'react';

export interface Column<Row> {
  /** Stable identity for the column, used as the React key. */
  key: string;
  /** Header text. */
  header: string;
  /** Cell contents for one row. */
  render: (row: Row) => ReactNode;
  /** Right-align and use tabular figures. Set for numeric columns. */
  numeric?: boolean;
  /** Marks the row's identifying cell, rendered as a `th` for screen readers. */
  rowHeader?: boolean;
}

interface DataTableProps<Row> {
  columns: Column<Row>[];
  rows: Row[];
  rowKey: (row: Row) => string;
  caption?: string;
}

/** A horizontally scrollable table with consistent styling. */
export function DataTable<Row>({ columns, rows, rowKey, caption }: DataTableProps<Row>) {
  return (
    <div className="table-scroll">
      <table className="table">
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                scope="col"
                key={column.key}
                className={column.numeric ? 'table__number' : undefined}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={rowKey(row)}>
              {columns.map((column) =>
                column.rowHeader ? (
                  <th scope="row" key={column.key}>
                    {column.render(row)}
                  </th>
                ) : (
                  <td key={column.key} className={column.numeric ? 'table__number' : undefined}>
                    {column.render(row)}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * A two-column name/value table, for parameters and tags.
 *
 * Separate from DataTable because the shape is fixed and the caller should not
 * have to describe two columns every time.
 */
export function KeyValueTable({
  entries,
  caption,
}: {
  entries: [string, ReactNode][];
  caption?: string;
}) {
  return (
    <div className="table-scroll">
      <table className="table">
        {caption && <caption>{caption}</caption>}
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <th scope="row">{key}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
