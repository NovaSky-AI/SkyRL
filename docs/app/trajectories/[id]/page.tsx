'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';

interface ConversationMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  text: string | null;
  position: number;
  has_image: boolean;
}

interface Trajectory {
  session_id: string;
  env_key: string;
  task_key: string;
  model: string;
  score: number;
  outcome: string;
  num_turns: number;
  num_screenshots: number;
  conversation: ConversationMessage[];
  image_paths: string[];
}

interface Turn {
  index: number;
  screenshot_path: string | null;
  assistant_text: string | null;
  tool_text: string | null;
}

function parseToolCall(text: string): string {
  try {
    const json = JSON.parse(text.replace(/<\/tool_call>$/, '').trim());
    return JSON.stringify(json, null, 2);
  } catch {
    return text;
  }
}

function groupIntoTurns(conv: ConversationMessage[], image_paths: string[]): Turn[] {
  const turns: Turn[] = [];
  let imageIndex = 0;

  // Skip system message (position 0)
  // Position 1 is the initial user message with task + first screenshot
  // Then groups of: assistant → tool → user(screenshot) → ...

  let i = 0;
  while (i < conv.length) {
    const msg = conv[i];

    if (msg.role === 'system') {
      i++;
      continue;
    }

    if (msg.role === 'user' && msg.has_image) {
      // This user message has a screenshot — it's the start of a turn
      const turn: Turn = {
        index: turns.length,
        screenshot_path: image_paths[imageIndex] ?? null,
        assistant_text: null,
        tool_text: null,
      };
      imageIndex++;

      // Look ahead for assistant + tool messages
      let j = i + 1;
      if (j < conv.length && conv[j].role === 'assistant') {
        turn.assistant_text = conv[j].text;
        j++;
      }
      if (j < conv.length && conv[j].role === 'tool') {
        turn.tool_text = conv[j].text;
        j++;
      }
      turns.push(turn);
      i = j;
      continue;
    }

    i++;
  }

  return turns;
}

export default function TrajectoryDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const [traj, setTraj] = useState<Trajectory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedTurn, setExpandedTurn] = useState<number | null>(0);

  useEffect(() => {
    fetch(`/api/trajectories/${id}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.error) { setError(data.error); } else { setTraj(data); }
        setLoading(false);
      })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, [id]);

  if (loading) return <div className="min-h-screen bg-gray-950 text-gray-100 p-6">Loading...</div>;
  if (error) return <div className="min-h-screen bg-gray-950 text-red-400 p-6">Error: {error}</div>;
  if (!traj) return null;

  const taskText = traj.conversation.find((m) => m.role === 'user' && m.text)?.text ?? '';
  const turns = groupIntoTurns(traj.conversation, traj.image_paths ?? []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-1">
              <Link href="/trajectories" className="text-gray-400 hover:text-gray-200 text-sm">← Back</Link>
              <span className="bg-gray-800 px-2 py-0.5 rounded text-xs font-mono">{traj.env_key}</span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                traj.outcome === 'pass'
                  ? 'bg-green-900/50 text-green-400'
                  : 'bg-red-900/50 text-red-400'
              }`}>{traj.outcome}</span>
              <span className="text-gray-400 text-xs">score: {traj.score.toFixed(2)}</span>
              <span className="text-gray-400 text-xs">{traj.num_turns} turns</span>
            </div>
            <div className="text-xs text-gray-500 font-mono truncate">{traj.session_id}</div>
          </div>
          <div className="text-xs text-gray-500 font-mono shrink-0">{traj.model}</div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 flex gap-6">
        {/* Task sidebar */}
        <div className="w-80 shrink-0">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 sticky top-20">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">Task</h2>
            <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{taskText}</p>
            <div className="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-500 font-mono break-all">{traj.task_key}</div>
          </div>
        </div>

        {/* Turns */}
        <div className="flex-1 min-w-0 space-y-3">
          {turns.length === 0 && (
            <div className="text-gray-500 text-sm">No turns found in this trajectory.</div>
          )}
          {turns.map((turn) => (
            <div key={turn.index} className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
              {/* Turn header */}
              <button
                onClick={() => setExpandedTurn(expandedTurn === turn.index ? null : turn.index)}
                className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-800/50"
              >
                <span className="text-gray-500 text-xs font-mono w-8">#{turn.index + 1}</span>
                <div className="flex-1 flex items-center gap-2 text-sm min-w-0">
                  {turn.screenshot_path ? (
                    <span className="text-blue-400 text-xs">📸 screenshot</span>
                  ) : (
                    <span className="text-gray-600 text-xs">no screenshot</span>
                  )}
                  {turn.assistant_text ? (
                    <span className="text-green-400 text-xs truncate font-mono">
                      {turn.assistant_text.slice(0, 80).replace(/\n/g, ' ')}
                    </span>
                  ) : (
                    <span className="text-yellow-600 text-xs">⚠ action not recorded</span>
                  )}
                </div>
                <span className="text-gray-600 text-xs shrink-0">{expandedTurn === turn.index ? '▲' : '▼'}</span>
              </button>

              {/* Turn body */}
              {expandedTurn === turn.index && (
                <div className="border-t border-gray-800">
                  <div className="flex gap-4 p-4">
                    {/* Screenshot */}
                    <div className="shrink-0">
                      {turn.screenshot_path ? (
                        <a
                          href={`/api/images/${turn.screenshot_path.replace(/^images\//, '')}`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={`/api/images/${turn.screenshot_path.replace(/^images\//, '')}`}
                            alt={`Step ${turn.index + 1}`}
                            className="w-72 rounded border border-gray-700 hover:opacity-90 cursor-zoom-in"
                          />
                        </a>
                      ) : (
                        <div className="w-72 h-40 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-600 text-sm">
                          No image
                        </div>
                      )}
                    </div>

                    {/* Text content */}
                    <div className="flex-1 min-w-0 space-y-4">
                      {/* Agent action */}
                      <div>
                        <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1.5">
                          Agent Action
                        </div>
                        {turn.assistant_text ? (
                          <pre className="bg-gray-950 border border-gray-700 rounded p-3 text-xs text-green-300 font-mono whitespace-pre-wrap break-all overflow-auto max-h-64">
                            {parseToolCall(turn.assistant_text)}
                          </pre>
                        ) : (
                          <div className="bg-yellow-900/20 border border-yellow-700/40 rounded p-3 text-xs text-yellow-500">
                            ⚠ Action not recorded in this dataset export
                          </div>
                        )}
                      </div>

                      {/* Tool result */}
                      {turn.tool_text && (
                        <div>
                          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1.5">
                            Result
                          </div>
                          <div className="bg-gray-950 border border-gray-700 rounded p-3 text-xs text-gray-300 font-mono">
                            {turn.tool_text}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
