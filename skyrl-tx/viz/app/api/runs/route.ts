import { getDb, Run } from "@/lib/db";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    const db = getDb();
    const runs = db.prepare("SELECT * FROM training_runs ORDER BY started_at DESC").all() as Run[];
    db.close();
    return NextResponse.json(runs);
  } catch (error) {
    console.error("Runs API error:", error);
    return NextResponse.json([]);
  }
}
