#!/bin/bash
# Runs as root in each sandbox before the agent (LabelledDaytonaEnvironment's setup_script).
#
# A Daytona sandbox reports the host's CPUs (64) to Node, so Jest, run by the agent without --maxWorkers,
# starts a worker per host CPU and the 4 GiB sandbox runs out of memory: the kill takes the agent with it.
# This preload makes os.cpus() / os.availableParallelism() report the sandbox's own CPU quota (cgroup),
# and the agent config loads it into every node process (NODE_OPTIONS in mini_swe_agent_textbased.yaml).
set -euo pipefail
mkdir -p /opt/skyrl
cat > /opt/skyrl/node_cpus.js <<'JS'
// The sandbox's CPUs, not the host's (see sandbox_setup.sh).
var os = require("os");
var fs = require("fs");
function quota() {
  try {
    var max = fs.readFileSync("/sys/fs/cgroup/cpu.max", "utf8").trim().split(/\s+/);
    if (max[0] !== "max") return Math.ceil(Number(max[0]) / Number(max[1]));
  } catch (e) {}
  try {
    var q = Number(fs.readFileSync("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "utf8"));
    var p = Number(fs.readFileSync("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "utf8"));
    if (q > 0 && p > 0) return Math.ceil(q / p);
  } catch (e) {}
  return Number(process.env.SKYRL_CPUS || 2);
}
var n = Math.max(1, Math.min(quota(), os.cpus().length));
var cpus = os.cpus().slice(0, n);
os.cpus = function () { return cpus; };
if (typeof os.availableParallelism === "function") os.availableParallelism = function () { return n; };
JS
chmod 644 /opt/skyrl/node_cpus.js
