import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import readline from 'readline';

const TRAJECTORIES_PATH = path.join(process.cwd(), '..', 'trajectories.jsonl');

async function findTrajectory(sessionId: string) {
  const rl = readline.createInterface({
    input: fs.createReadStream(TRAJECTORIES_PATH),
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const traj = JSON.parse(line);
    if (traj.session_id === sessionId) {
      rl.close();
      return traj;
    }
  }
  return null;
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const traj = await findTrajectory(id);
  if (!traj) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }
  // Parse conversation from JSON string
  if (typeof traj.conversation === 'string') {
    traj.conversation = JSON.parse(traj.conversation);
  }
  return NextResponse.json(traj);
}
