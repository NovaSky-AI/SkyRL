"use client";

import { useState, useEffect } from "react";
import type { Run, Step } from "@/lib/db";

export function useRunData(runId: string) {
  const [run, setRun] = useState<Run | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;

    setIsLoading(true);
    setError(null);

    fetch(`/api/runs/${runId}`)
      .then((r) => r.json())
      .then((data) => {
        setRun(data.run);
        setSteps(data.steps || []);
      })
      .catch((e) => {
        setError(e.message);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [runId]);

  return { run, steps, isLoading, error };
}
