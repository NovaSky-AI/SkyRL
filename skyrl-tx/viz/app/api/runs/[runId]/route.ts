import { getDb, Run, Step } from "@/lib/db";
import { NextResponse } from "next/server";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ runId: string }> }
) {
  try {
    const { runId } = await params;
    const db = getDb();

    const run = db.prepare("SELECT * FROM training_runs WHERE id = ?").get(runId) as Run | undefined;

    if (!run) {
      db.close();
      return NextResponse.json({ error: "Run not found" }, { status: 404 });
    }

    const steps = db
      .prepare(
        `
        SELECT
          id, run_id, step, created_at as timestamp,
          loss, reward_mean, reward_std, kl_divergence, entropy, learning_rate,
          ac_tokens_per_turn, ob_tokens_per_turn,
          total_ac_tokens, total_turns,
          sampling_time_mean, time_total,
          frac_mixed, frac_all_good, frac_all_bad, extras
        FROM training_steps
        WHERE run_id = ?
        ORDER BY step
      `
      )
      .all(runId) as Step[];

    db.close();

    return NextResponse.json({ run, steps });
  } catch (error) {
    console.error("Run API error:", error);
    return NextResponse.json({ error: "Database error" }, { status: 500 });
  }
}
