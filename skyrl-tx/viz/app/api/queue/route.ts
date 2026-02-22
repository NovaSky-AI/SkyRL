import { getTinkerDb, TinkerFuture, RequestType, QueueStats } from "@/lib/db";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const REQUEST_TYPES: RequestType[] = [
  "CREATE_MODEL",
  "FORWARD_BACKWARD",
  "FORWARD",
  "OPTIM_STEP",
  "SAVE_WEIGHTS",
  "SAVE_WEIGHTS_FOR_SAMPLER",
  "LOAD_WEIGHTS",
  "SAMPLE",
  "EXTERNAL",
];

export async function GET() {
  try {
    const db = getTinkerDb();
    if (!db) {
      return NextResponse.json({ available: false, stats: null, recentRequests: [] });
    }

    const stats = db.prepare(`
      SELECT status, COUNT(*) as count FROM futures GROUP BY status
    `).all() as { status: string; count: number }[];

    const statsMap: Record<string, number> = {};
    for (const row of stats) {
      statsMap[row.status] = row.count;
    }

    const byTypeRaw = db.prepare(`
      SELECT request_type, status, COUNT(*) as count
      FROM futures GROUP BY request_type, status
    `).all() as { request_type: RequestType; status: string; count: number }[];

    const byType: QueueStats["byType"] = {} as QueueStats["byType"];
    for (const type of REQUEST_TYPES) {
      byType[type] = { pending: 0, completed: 0, failed: 0 };
    }
    for (const row of byTypeRaw) {
      if (byType[row.request_type]) {
        byType[row.request_type][row.status as "pending" | "completed" | "failed"] = row.count;
      }
    }

    const recentRequests = db.prepare(`
      SELECT request_id, request_type, model_id, status, created_at, completed_at
      FROM futures ORDER BY request_id DESC LIMIT 50
    `).all() as Omit<TinkerFuture, "request_data" | "result_data">[];

    const latencyStats = db.prepare(`
      SELECT
        AVG((julianday(completed_at) - julianday(created_at)) * 86400) as avg_latency_seconds,
        MIN((julianday(completed_at) - julianday(created_at)) * 86400) as min_latency_seconds,
        MAX((julianday(completed_at) - julianday(created_at)) * 86400) as max_latency_seconds
      FROM futures WHERE status = 'completed' AND completed_at IS NOT NULL
    `).get() as {
      avg_latency_seconds: number | null;
      min_latency_seconds: number | null;
      max_latency_seconds: number | null;
    };

    db.close();

    return NextResponse.json({
      available: true,
      stats: {
        pending: statsMap["pending"] || 0,
        completed: statsMap["completed"] || 0,
        failed: statsMap["failed"] || 0,
        byType,
      } as QueueStats,
      latency: latencyStats,
      recentRequests,
    });
  } catch (error) {
    console.error("Queue API error:", error);
    return NextResponse.json({ available: false, stats: null, recentRequests: [], error: String(error) });
  }
}
