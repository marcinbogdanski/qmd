#!/usr/bin/env bun
/**
 * QMD Query Expansion Benchmark
 *
 * Measures query expansion latency across different query types.
 * Reports per-query latency, result type breakdown (lex/vec/hyde), and peak RSS.
 *
 * Usage:
 *   bun bench/bench-expand.ts           # full benchmark
 *   bun bench/bench-expand.ts --quick   # quick smoke test
 */

import { LlamaCpp, type Queryable } from "../src/llm.ts";
import { heapStats } from "bun:jsc";

// ============================================================================
// Config
// ============================================================================

const args = process.argv.slice(2);
const quick = args.includes("--quick");

const QUERIES: { label: string; text: string }[] = [
  { label: "short",    text: "machine learning" },
  { label: "question", text: "How do transformers handle long-range dependencies?" },
  { label: "complex",  text: "compare BM25 sparse retrieval vs dense vector search for domain-specific corpora" },
  { label: "code",     text: "async function embedBatch parallel context splitting" },
  { label: "natural",  text: "I want to find my notes about the quarterly review from last month" },
];

const TEST_QUERIES = quick ? QUERIES.slice(0, 2) : QUERIES;
const ITERATIONS = quick ? 1 : 3;

// ============================================================================
// Helpers
// ============================================================================

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function typeBreakdown(queryables: Queryable[]): string {
  const counts: Record<string, number> = {};
  for (const q of queryables) counts[q.type] = (counts[q.type] || 0) + 1;
  return Object.entries(counts).map(([t, n]) => `${t}:${n}`).join(" ");
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("  QMD Query Expansion Benchmark");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const llm = new LlamaCpp();

  // System info
  const device = await llm.getDeviceInfo();
  console.log("System");
  console.log(`  Device:    ${device.gpu || "cpu"}`);
  if (device.gpuDevices.length > 0) console.log(`  GPU:       ${device.gpuDevices.join(", ")}`);
  if (device.vram) console.log(`  VRAM:      ${formatBytes(device.vram.total)} total, ${formatBytes(device.vram.free)} free`);
  console.log(`  CPU cores: ${device.cpuCores} math`);
  console.log(`  RSS:       ${formatBytes(process.memoryUsage().rss)} at start`);

  // Cold-start: first expandQuery triggers model load
  console.log("\nLoading model...");
  const initT0 = performance.now();
  await llm.expandQuery("warmup query");
  const initMs = performance.now() - initT0;
  console.log(`  Model loaded in ${initMs.toFixed(0)}ms (fresh context per call)`);

  // Benchmark
  console.log(`\nBenchmark (1 cold + ${ITERATIONS} warm iteration${ITERATIONS > 1 ? "s" : ""} per query)\n`);

  type Result = {
    label: string;
    query: string;
    coldMs: number;
    warmMs: number;
    breakdown: string;
    totalResults: number;
    peakRss: number;
  };
  const results: Result[] = [];
  let peakRss = process.memoryUsage().rss;

  for (const { label, text } of TEST_QUERIES) {

    process.stdout.write(`  [${label.padEnd(8)}] `);

    // Cold run (first invocation for this query)
    const t0 = performance.now();
    const coldResult = await llm.expandQuery(text);
    const coldMs = performance.now() - t0;

    let rss = process.memoryUsage().rss;
    if (rss > peakRss) peakRss = rss;

    // Warm runs
    const warmTimes: number[] = [];
    let lastResult = coldResult;
    for (let iter = 0; iter < ITERATIONS; iter++) {
      const t1 = performance.now();
      lastResult = await llm.expandQuery(text);
      warmTimes.push(performance.now() - t1);

      rss = process.memoryUsage().rss;
      if (rss > peakRss) peakRss = rss;
    }

    const warmMs = median(warmTimes);
    const breakdown = typeBreakdown(lastResult);
    results.push({ label, query: text, coldMs, warmMs, breakdown, totalResults: lastResult.length, peakRss });
    console.log(`cold ${coldMs.toFixed(0).padStart(5)}ms  warm ${warmMs.toFixed(0).padStart(5)}ms  ${String(lastResult.length).padStart(2)} results  [${breakdown}]`);
  }

  // Heap stats
  Bun.gc(true);
  const heap = heapStats();
  console.log(`\nHeap: ${formatBytes(heap.heapSize)} (${heap.objectCount} objects)`);

  // Summary
  console.log("\n═══════════════════════════════════════════════════════════════");
  console.log("  Results");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const header = "  Type        Cold      Warm  Results  Breakdown         Peak RSS";
  const sep    = "  ────────  ──────    ──────  ───────  ────────────────  ────────";
  console.log(header);
  console.log(sep);

  for (const r of results) {
    console.log(
      `  ${r.label.padEnd(8)}  ` +
      `${r.coldMs.toFixed(0).padStart(5)}ms  ` +
      `${r.warmMs.toFixed(0).padStart(5)}ms  ` +
      `${String(r.totalResults).padStart(7)}  ` +
      `${r.breakdown.padEnd(16)}  ` +
      `${formatBytes(r.peakRss).padStart(8)}`
    );
  }

  const avgWarm = results.reduce((s, r) => s + r.warmMs, 0) / results.length;
  const avgCold = results.reduce((s, r) => s + r.coldMs, 0) / results.length;
  console.log(`\n  Average: cold ${avgCold.toFixed(0)}ms, warm ${avgWarm.toFixed(0)}ms per query`);

  console.log("");
  await llm.dispose();
}

main().catch(console.error);
