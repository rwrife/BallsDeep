// src/data/csv.js
// Tiny CSV parser. Handles quoted fields, escaped quotes, \r\n, basic UTF-8.
// Returns { headers, rows } where rows are arrays of strings.

export function parseCSV(text, opts = {}) {
  const delim = opts.delimiter || ',';
  const rows = [];
  let row = [], field = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === delim) { row.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = '';
        if (row.length > 1 || row[0] !== '') rows.push(row);
        row = [];
      } else field += c;
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  if (rows.length === 0) return { headers: [], rows: [] };
  const headers = opts.headers === false ? null : rows.shift();
  return { headers, rows };
}

/** Convert parsed CSV rows to numeric array of arrays.
 *  Non-numeric / blank cells become `naFill` (default 0). */
export function csvToNumeric(rows, naFill = 0) {
  return rows.map((r) => r.map((v) => {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : naFill;
  }));
}

/** Convenience: load a numeric (X, y) split from CSV text given a label column. */
export function csvToXY(text, labelCol, opts = {}) {
  const { headers, rows } = parseCSV(text, opts);
  if (!headers) throw new Error('csvToXY requires headers');
  const labelIdx = typeof labelCol === 'number'
    ? labelCol
    : headers.indexOf(labelCol);
  if (labelIdx < 0) throw new Error(`label column "${labelCol}" not found`);
  const featureNames = headers.filter((_, i) => i !== labelIdx);
  const X = [], y = [];
  for (const r of rows) {
    const yv = parseFloat(r[labelIdx]);
    y.push(Number.isFinite(yv) ? yv : 0);
    const xr = [];
    for (let i = 0; i < r.length; i++) {
      if (i === labelIdx) continue;
      const v = parseFloat(r[i]);
      xr.push(Number.isFinite(v) ? v : 0);
    }
    X.push(xr);
  }
  return { X, y, featureNames };
}

/** Node-only convenience to load a CSV file from disk. */
export async function loadCSVFile(path, opts) {
  const fs = await import('node:fs/promises');
  const text = await fs.readFile(path, 'utf8');
  return parseCSV(text, opts);
}
