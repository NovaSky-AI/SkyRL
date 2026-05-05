import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const IMAGES_BASE = path.join(process.cwd(), '..', 'fleet-cu-trajectories', 'images');

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: segments } = await params;
  const filePath = path.join(IMAGES_BASE, ...segments);

  // Prevent path traversal
  if (!filePath.startsWith(IMAGES_BASE)) {
    return new NextResponse('Forbidden', { status: 403 });
  }

  if (!fs.existsSync(filePath)) {
    return new NextResponse('Not found', { status: 404 });
  }

  const data = fs.readFileSync(filePath);
  return new NextResponse(data, {
    headers: { 'Content-Type': 'image/jpeg', 'Cache-Control': 'public, max-age=86400' },
  });
}
