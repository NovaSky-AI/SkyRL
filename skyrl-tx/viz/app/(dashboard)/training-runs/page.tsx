"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import type { Run } from "@/lib/db";

export default function TrainingRunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/runs", { cache: "no-store" })
      .then((r) => r.json())
      .then(setRuns)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-6">Training runs</h1>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>NAME</TableHead>
            <TableHead>TYPE</TableHead>
            <TableHead>MODALITY</TableHead>
            <TableHead>STARTED</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
                Loading...
              </TableCell>
            </TableRow>
          ) : runs.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
                No training runs yet.
              </TableCell>
            </TableRow>
          ) : (
            runs.map((run) => (
              <TableRow key={run.id}>
                <TableCell className="font-mono text-sm">
                  <Link href={`/training-run/${run.id}`} className="hover:underline text-blue-600">
                    {run.id}
                  </Link>
                </TableCell>
                <TableCell>{run.name}</TableCell>
                <TableCell>
                  <Badge variant="outline" className="uppercase">{run.type}</Badge>
                </TableCell>
                <TableCell>
                  <Badge variant={run.modality === "vision" ? "purple" : "secondary"} className="uppercase">
                    {run.modality}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {run.started_at ? new Date(run.started_at).toLocaleString() : "N/A"}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
