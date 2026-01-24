import { getTinkerDb, TinkerCheckpoint } from "@/lib/db";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const modelId = searchParams.get("model_id");

    const db = getTinkerDb();

    if (!db) {
      return NextResponse.json({
        available: false,
        checkpoints: [],
      });
    }

    let query = `
      SELECT
        c.model_id,
        c.checkpoint_id,
        c.checkpoint_type,
        c.status,
        c.created_at,
        c.completed_at,
        c.error_message,
        m.base_model
      FROM checkpoints c
      LEFT JOIN models m ON c.model_id = m.model_id
    `;

    const params: string[] = [];

    if (modelId) {
      query += " WHERE c.model_id = ?";
      params.push(modelId);
    }

    query += " ORDER BY c.created_at DESC LIMIT 100";

    const checkpoints = db.prepare(query).all(...params) as (TinkerCheckpoint & {
      base_model: string | null;
    })[];

    // Get checkpoint stats
    const statsQuery = modelId
      ? `
        SELECT
          checkpoint_type,
          status,
          COUNT(*) as count
        FROM checkpoints
        WHERE model_id = ?
        GROUP BY checkpoint_type, status
      `
      : `
        SELECT
          checkpoint_type,
          status,
          COUNT(*) as count
        FROM checkpoints
        GROUP BY checkpoint_type, status
      `;

    const statsRaw = modelId
      ? db.prepare(statsQuery).all(modelId) as { checkpoint_type: string; status: string; count: number }[]
      : db.prepare(statsQuery).all() as { checkpoint_type: string; status: string; count: number }[];

    const stats = {
      training: { pending: 0, completed: 0, failed: 0 },
      sampler: { pending: 0, completed: 0, failed: 0 },
    };

    for (const row of statsRaw) {
      const type = row.checkpoint_type === "TRAINING" ? "training" : "sampler";
      const status = row.status as "pending" | "completed" | "failed";
      stats[type][status] = row.count;
    }

    db.close();

    return NextResponse.json({
      available: true,
      checkpoints,
      stats,
    });
  } catch (error) {
    console.error("Checkpoints API error:", error);
    return NextResponse.json({
      available: false,
      checkpoints: [],
      error: String(error),
    });
  }
}
