'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';

interface TrajectorySummary {
  session_id: string;
  env_key: string;
  task_key: string;
  model: string;
  score: number;
  outcome: string;
  num_turns: number;
  num_screenshots: number;
  has_images: boolean;
}

interface ApiResponse {
  total: number;
  page: number;
  limit: number;
  trajectories: TrajectorySummary[];
}

export default function TrajectoriesPage() {
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [envFilter, setEnvFilter] = useState('');
  const [outcomeFilter, setOutcomeFilter] = useState('');
  const [imagesOnly, setImagesOnly] = useState(false);
  const limit = 50;

  const fetchData = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    if (envFilter) params.set('env', envFilter);
    if (outcomeFilter) params.set('outcome', outcomeFilter);
    if (imagesOnly) params.set('has_images', 'true');
    const res = await fetch(`/api/trajectories?${params}`);
    const json = await res.json();
    setData(json);
    setLoading(false);
  }, [page, envFilter, outcomeFilter, imagesOnly]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const totalPages = data ? Math.ceil(data.total / limit) : 1;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Trajectory Viewer</h1>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 mb-6">
          <input
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm w-40"
            placeholder="Filter by env..."
            value={envFilter}
            onChange={(e) => { setEnvFilter(e.target.value); setPage(1); }}
          />
          <select
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
            value={outcomeFilter}
            onChange={(e) => { setOutcomeFilter(e.target.value); setPage(1); }}
          >
            <option value="">All outcomes</option>
            <option value="pass">pass</option>
            <option value="fail">fail</option>
          </select>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={imagesOnly}
              onChange={(e) => { setImagesOnly(e.target.checked); setPage(1); }}
              className="rounded"
            />
            Has images only
          </label>
          {data && (
            <span className="text-gray-400 text-sm self-center">
              {data.total.toLocaleString()} trajectories
            </span>
          )}
        </div>

        {loading ? (
          <div className="text-gray-400">Loading...</div>
        ) : (
          <>
            <div className="overflow-x-auto rounded-lg border border-gray-800">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-900 text-gray-400 text-left">
                    <th className="px-4 py-3 font-medium">Env</th>
                    <th className="px-4 py-3 font-medium">Task</th>
                    <th className="px-4 py-3 font-medium">Model</th>
                    <th className="px-4 py-3 font-medium">Outcome</th>
                    <th className="px-4 py-3 font-medium">Score</th>
                    <th className="px-4 py-3 font-medium">Turns</th>
                    <th className="px-4 py-3 font-medium">Images</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.trajectories.map((t) => (
                    <tr
                      key={t.session_id}
                      className="border-t border-gray-800 hover:bg-gray-900/50"
                    >
                      <td className="px-4 py-2.5">
                        <span className="bg-gray-800 px-2 py-0.5 rounded text-xs font-mono">{t.env_key}</span>
                      </td>
                      <td className="px-4 py-2.5 max-w-xs">
                        <Link
                          href={`/trajectories/${t.session_id}`}
                          className="text-blue-400 hover:text-blue-300 font-mono text-xs truncate block"
                        >
                          {t.task_key}
                        </Link>
                      </td>
                      <td className="px-4 py-2.5 text-gray-300 text-xs font-mono">{t.model}</td>
                      <td className="px-4 py-2.5">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          t.outcome === 'pass'
                            ? 'bg-green-900/50 text-green-400'
                            : 'bg-red-900/50 text-red-400'
                        }`}>
                          {t.outcome}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-gray-300">{t.score.toFixed(2)}</td>
                      <td className="px-4 py-2.5 text-gray-300">{t.num_turns}</td>
                      <td className="px-4 py-2.5">
                        {t.has_images ? (
                          <span className="text-green-400 text-xs">✓</span>
                        ) : (
                          <span className="text-gray-600 text-xs">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-4">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-4 py-2 bg-gray-800 rounded disabled:opacity-40 text-sm hover:bg-gray-700"
              >
                Previous
              </button>
              <span className="text-gray-400 text-sm">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="px-4 py-2 bg-gray-800 rounded disabled:opacity-40 text-sm hover:bg-gray-700"
              >
                Next
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
