import { getDb } from "@/lib/db";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    const db = getDb();

    const totalRuns = db.prepare("SELECT COUNT(*) as count FROM training_runs").get() as { count: number };
    const runningRuns = db.prepare("SELECT COUNT(*) as count FROM training_runs WHERE ended_at IS NULL").get() as { count: number };

    const runs = db.prepare(`
      SELECT id, name, config, started_at, ended_at
      FROM training_runs
      ORDER BY started_at DESC
    `).all() as { id: string; name: string; config: string | null; started_at: string; ended_at: string | null }[];

    const runsWithModel = runs.map(run => {
      let model = "Unknown";
      if (run.config) {
        try {
          const config = JSON.parse(run.config);
          model = config.model || config.model_name || config.base_model || "Unknown";
          if (model.includes("/")) {
            model = model.split("/").pop() || model;
          }
        } catch {}
      }
      return { ...run, model };
    });

    const perfStats = db.prepare(`
      SELECT
        AVG(ac_tokens_per_turn) as avg_tokens_per_turn,
        SUM(total_ac_tokens) as total_action_tokens,
        SUM(total_turns) as total_turns,
        AVG(sampling_time_mean) as avg_sampling_time,
        AVG(time_total) as avg_step_time
      FROM training_steps
      WHERE ac_tokens_per_turn IS NOT NULL
         OR total_ac_tokens IS NOT NULL
    `).get() as {
      avg_tokens_per_turn: number | null;
      total_action_tokens: number | null;
      total_turns: number | null;
      avg_sampling_time: number | null;
      avg_step_time: number | null;
    };

    const perfOverTime = db.prepare(`
      SELECT
        date(created_at) as day,
        AVG(ac_tokens_per_turn) as tokens_per_turn,
        AVG(total_ac_tokens) as tokens_per_step,
        COUNT(*) as num_steps
      FROM training_steps
      WHERE total_ac_tokens IS NOT NULL
      GROUP BY date(created_at)
      ORDER BY day ASC
      LIMIT 30
    `).all() as { day: string; tokens_per_turn: number | null; tokens_per_step: number | null; num_steps: number }[];

    const rewardOverTime = db.prepare(`
      SELECT
        date(created_at) as day,
        AVG(reward_mean) as avg_reward,
        MAX(reward_mean) as max_reward
      FROM training_steps
      WHERE reward_mean IS NOT NULL
      GROUP BY date(created_at)
      ORDER BY day ASC
      LIMIT 30
    `).all() as { day: string; avg_reward: number | null; max_reward: number | null }[];

    db.close();

    return NextResponse.json({
      totalRuns: totalRuns.count,
      runningRuns: runningRuns.count,
      runs: runsWithModel.slice(0, 10),
      perfStats: {
        avgTokensPerTurn: perfStats.avg_tokens_per_turn,
        totalActionTokens: perfStats.total_action_tokens,
        totalTurns: perfStats.total_turns,
        avgSamplingTime: perfStats.avg_sampling_time,
        avgStepTime: perfStats.avg_step_time,
      },
      perfOverTime,
      rewardOverTime,
    });
  } catch (error) {
    console.error("Stats API error:", error);
    return NextResponse.json({
      totalRuns: 0,
      runningRuns: 0,
      runs: [],
      perfStats: {
        avgTokensPerTurn: null,
        totalActionTokens: null,
        totalTurns: null,
        avgSamplingTime: null,
        avgStepTime: null,
      },
      perfOverTime: [],
      rewardOverTime: [],
    });
  }
}
