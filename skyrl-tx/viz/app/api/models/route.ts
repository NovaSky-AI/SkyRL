import { getTinkerDb, TinkerModel } from "@/lib/db";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    const db = getTinkerDb();
    if (!db) {
      return NextResponse.json({ available: false, models: [] });
    }

    const models = db.prepare(`
      SELECT m.model_id, m.base_model, m.lora_config, m.status, m.session_id, m.created_at,
             s.tags as session_tags, s.status as session_status
      FROM models m
      LEFT JOIN sessions s ON m.session_id = s.session_id
      ORDER BY m.created_at DESC
    `).all() as (TinkerModel & { session_tags: string | null; session_status: string | null })[];

    const checkpointCounts = db.prepare(`
      SELECT model_id, checkpoint_type, COUNT(*) as count,
             SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count
      FROM checkpoints GROUP BY model_id, checkpoint_type
    `).all() as { model_id: string; checkpoint_type: string; count: number; completed_count: number }[];

    const checkpointMap: Record<string, { training: number; sampler: number }> = {};
    for (const row of checkpointCounts) {
      if (!checkpointMap[row.model_id]) {
        checkpointMap[row.model_id] = { training: 0, sampler: 0 };
      }
      if (row.checkpoint_type === "TRAINING") {
        checkpointMap[row.model_id].training = row.completed_count;
      } else if (row.checkpoint_type === "SAMPLER") {
        checkpointMap[row.model_id].sampler = row.completed_count;
      }
    }

    const requestCounts = db.prepare(`
      SELECT model_id, COUNT(*) as total_requests,
             SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_requests,
             SUM(CASE WHEN request_type = 'FORWARD_BACKWARD' THEN 1 ELSE 0 END) as training_steps
      FROM futures WHERE model_id IS NOT NULL GROUP BY model_id
    `).all() as { model_id: string; total_requests: number; completed_requests: number; training_steps: number }[];

    const requestMap: Record<string, { total: number; completed: number; trainingSteps: number }> = {};
    for (const row of requestCounts) {
      requestMap[row.model_id] = {
        total: row.total_requests,
        completed: row.completed_requests,
        trainingSteps: row.training_steps,
      };
    }

    db.close();

    const enrichedModels = models.map((model) => {
      let loraConfig = null;
      try { loraConfig = JSON.parse(model.lora_config); } catch {}

      let sessionTags: string[] = [];
      try { if (model.session_tags) sessionTags = JSON.parse(model.session_tags); } catch {}

      return {
        ...model,
        lora_config: loraConfig,
        session_tags: sessionTags,
        checkpoints: checkpointMap[model.model_id] || { training: 0, sampler: 0 },
        requests: requestMap[model.model_id] || { total: 0, completed: 0, trainingSteps: 0 },
      };
    });

    return NextResponse.json({ available: true, models: enrichedModels });
  } catch (error) {
    console.error("Models API error:", error);
    return NextResponse.json({ available: false, models: [], error: String(error) });
  }
}
