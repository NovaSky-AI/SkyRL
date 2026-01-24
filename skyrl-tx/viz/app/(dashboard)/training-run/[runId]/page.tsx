"use client";

import { useState, useMemo } from "react";
import { useParams } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { useRunData } from "@/hooks/useRunData";
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  AreaChart,
  Area,
  LineChart,
  Line,
} from "recharts";

export default function TrainingRunPage() {
  const params = useParams();
  const runId = params.runId as string;
  const decodedRunId = decodeURIComponent(runId);

  const { run, steps, isLoading } = useRunData(decodedRunId);

  const config = run?.config ? JSON.parse(run.config) : {};
  const latest = steps[steps.length - 1];
  const isEnded = !!run?.ended_at;

  const chartData = useMemo(() => {
    return steps.map((s) => ({
      step: s.step,
      reward_mean: s.reward_mean,
      loss: s.loss,
      kl_divergence: s.kl_divergence,
      entropy: s.entropy,
      learning_rate: s.learning_rate,
      ac_tokens_per_turn: s.ac_tokens_per_turn,
      sampling_time_mean: s.sampling_time_mean,
      time_total: s.time_total,
    }));
  }, [steps]);

  if (isLoading) {
    return (
      <div className="p-4 max-w-[1400px] mx-auto">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-muted rounded w-1/3"></div>
          <div className="h-64 bg-muted rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4 max-w-[1400px] mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between pb-3 border-b border-border">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold font-mono">{run?.name || decodedRunId}</h1>
          <Badge variant={isEnded ? "secondary" : "default"}>
            {isEnded ? "Ended" : "Live"}
          </Badge>
          {run?.type && <Badge variant="outline" className="text-xs uppercase">{run.type}</Badge>}
        </div>

        {latest && (
          <div className="text-right">
            <div className="font-mono font-bold">Step {latest.step}</div>
            <div className="text-xs text-muted-foreground">
              reward: {latest.reward_mean?.toFixed(3) ?? "N/A"}
            </div>
          </div>
        )}
      </header>

      {/* Config */}
      {Object.keys(config).length > 0 && (
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm py-2 px-3 bg-muted/30 rounded-lg">
          {Object.entries(config)
            .filter(([k]) => {
              const essentialKeys = [
                "model_name", "model", "env_type", "batch_size", "group_size",
                "learning_rate", "lr", "max_steps", "max_tokens", "task", "hf_repo"
              ];
              return essentialKeys.includes(k);
            })
            .map(([k, v]) => (
              <div key={k}>
                <span className="text-muted-foreground">{k}:</span>
                <span className="font-mono ml-1">{String(v)}</span>
              </div>
            ))}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4">
        {steps.some(s => s.reward_mean !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">Reward Mean</div>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Area type="monotone" dataKey="reward_mean" stroke="#22c55e" fill="#22c55e33" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}

        {steps.some(s => s.loss !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">Loss</div>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Area type="monotone" dataKey="loss" stroke="#ef4444" fill="#ef444433" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}

        {steps.some(s => s.kl_divergence !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">KL Divergence</div>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Line type="monotone" dataKey="kl_divergence" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}

        {steps.some(s => s.entropy !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">Entropy</div>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Line type="monotone" dataKey="entropy" stroke="#a855f7" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}

        {steps.some(s => s.learning_rate !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">Learning Rate</div>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} tickFormatter={(v) => v.toExponential(1)} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Line type="monotone" dataKey="learning_rate" stroke="#f59e0b" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}

        {steps.some(s => s.ac_tokens_per_turn !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">Tokens per Turn</div>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Area type="monotone" dataKey="ac_tokens_per_turn" stroke="#14b8a6" fill="#14b8a633" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}

        {steps.some(s => s.sampling_time_mean !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">Sampling Time (s)</div>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Area type="monotone" dataKey="sampling_time_mean" stroke="#ec4899" fill="#ec489933" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}

        {steps.some(s => s.time_total !== null) && (
          <Card className="p-4">
            <div className="text-sm font-medium mb-2">Step Time (s)</div>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", fontSize: 12 }} />
                <Area type="monotone" dataKey="time_total" stroke="#6366f1" fill="#6366f133" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>

      {steps.length === 0 && (
        <Card className="p-8 text-center text-muted-foreground">
          No training data available yet
        </Card>
      )}
    </div>
  );
}
