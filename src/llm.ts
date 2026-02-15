/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */

import {
  getLlama,
  getLlamaGpuTypes,
  resolveModelFile,
  LlamaChatSession,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
  type LlamaContext,
  type LlamaEmbeddingContext,
  type LlamaGrammar,
  type Token as LlamaToken,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync, statSync, unlinkSync, readdirSync, readFileSync, writeFileSync } from "fs";

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields.
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Options for LLM sessions
 */
export type LLMSessionOptions = {
  /** Max session duration in ms (default: 10 minutes) */
  maxDuration?: number;
  /** External abort signal */
  signal?: AbortSignal;
  /** Debug name for logging */
  name?: string;
};

/**
 * Session interface for scoped LLM access with lifecycle guarantees
 */
export interface ILLMSession {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]>;
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
  /** Whether this session is still valid (not released or aborted) */
  readonly isValid: boolean;
  /** Abort signal for this session (aborts on release or maxDuration) */
  readonly signal: AbortSignal;
}

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// Model Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
// const DEFAULT_GENERATE_MODEL = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

export const DEFAULT_EMBED_MODEL_URI = DEFAULT_EMBED_MODEL;
export const DEFAULT_RERANK_MODEL_URI = DEFAULT_RERANK_MODEL;
export const DEFAULT_GENERATE_MODEL_URI = DEFAULT_GENERATE_MODEL;

// Local model cache directory
const MODEL_CACHE_DIR = join(homedir(), ".cache", "qmd", "models");
export const DEFAULT_MODEL_CACHE_DIR = MODEL_CACHE_DIR;

export type PullResult = {
  model: string;
  path: string;
  sizeBytes: number;
  refreshed: boolean;
};

type HfRef = {
  repo: string;
  file: string;
};

function parseHfUri(model: string): HfRef | null {
  if (!model.startsWith("hf:")) return null;
  const without = model.slice(3);
  const parts = without.split("/");
  if (parts.length < 3) return null;
  const repo = parts.slice(0, 2).join("/");
  const file = parts.slice(2).join("/");
  return { repo, file };
}

async function getRemoteEtag(ref: HfRef): Promise<string | null> {
  const url = `https://huggingface.co/${ref.repo}/resolve/main/${ref.file}`;
  try {
    const resp = await fetch(url, { method: "HEAD" });
    if (!resp.ok) return null;
    const etag = resp.headers.get("etag");
    return etag || null;
  } catch {
    return null;
  }
}

export async function pullModels(
  models: string[],
  options: { refresh?: boolean; cacheDir?: string } = {}
): Promise<PullResult[]> {
  const cacheDir = options.cacheDir || MODEL_CACHE_DIR;
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir, { recursive: true });
  }

  const results: PullResult[] = [];
  for (const model of models) {
    let refreshed = false;
    const hfRef = parseHfUri(model);
    const filename = model.split("/").pop();
    const entries = readdirSync(cacheDir, { withFileTypes: true });
    const cached = filename
      ? entries
          .filter((entry) => entry.isFile() && entry.name.includes(filename))
          .map((entry) => join(cacheDir, entry.name))
      : [];

    if (hfRef && filename) {
      const etagPath = join(cacheDir, `${filename}.etag`);
      const remoteEtag = await getRemoteEtag(hfRef);
      const localEtag = existsSync(etagPath)
        ? readFileSync(etagPath, "utf-8").trim()
        : null;
      const shouldRefresh =
        options.refresh || !remoteEtag || remoteEtag !== localEtag || cached.length === 0;

      if (shouldRefresh) {
        for (const candidate of cached) {
          if (existsSync(candidate)) unlinkSync(candidate);
        }
        if (existsSync(etagPath)) unlinkSync(etagPath);
        refreshed = cached.length > 0;
      }
    } else if (options.refresh && filename) {
      for (const candidate of cached) {
        if (existsSync(candidate)) unlinkSync(candidate);
        refreshed = true;
      }
    }

    const path = await resolveModelFile(model, cacheDir);
    const sizeBytes = existsSync(path) ? statSync(path).size : 0;
    if (hfRef && filename) {
      const remoteEtag = await getRemoteEtag(hfRef);
      if (remoteEtag) {
        const etagPath = join(cacheDir, `${filename}.etag`);
        writeFileSync(etagPath, remoteEtag + "\n", "utf-8");
      }
    }
    results.push({ model, path, sizeBytes, refreshed });
  }
  return results;
}

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// IdleResourceSlot - LRU/idle-cache for lazy GPU resources
// =============================================================================

type IdleResourceSlotConfig<T> = {
  name: string;
  create: () => Promise<T>;
  dispose: (resource: T) => Promise<void>;
  idleTimeoutMs: number;
  canEvict?: () => boolean;
};

/**
 * Generic lazy-create + idle-evict slot for GPU resources.
 *
 * Encapsulates: lazy creation, promise dedup, per-slot idle timeout, disposal.
 * Each slot manages its own lifecycle independently — resources that haven't been
 * used recently are evicted without affecting other resource groups.
 */
// Exported for testing only — allows tests to inspect/manipulate slot internals.
// Not part of the public API.
export class IdleResourceSlot<T> {
  readonly config: IdleResourceSlotConfig<T>;
  private resource: T | null = null;
  private createPromise: Promise<T> | null = null;
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private _disposed = false;

  constructor(config: IdleResourceSlotConfig<T>) {
    this.config = config;
  }

  /** Whether the resource is currently loaded (without triggering creation). */
  get isLoaded(): boolean {
    return this.resource !== null;
  }

  /** Peek at the current resource without triggering creation or touching the timer. */
  get current(): T | null {
    return this.resource;
  }

  /** Get the resource, creating it lazily if needed. Resets idle timer. */
  async get(): Promise<T> {
    if (this._disposed) {
      throw new Error(`IdleResourceSlot "${this.config.name}" is disposed`);
    }
    if (this.resource !== null) {
      this.touch();
      return this.resource;
    }
    if (this.createPromise) {
      return this.createPromise;
    }
    this.createPromise = (async () => {
      const r = await this.config.create();
      if (this._disposed) {
        await this.config.dispose(r);
        throw new Error(`IdleResourceSlot "${this.config.name}" was disposed during creation`);
      }
      this.resource = r;
      this.touch();
      return r;
    })();
    try {
      return await this.createPromise;
    } finally {
      this.createPromise = null;
    }
  }

  /** Reset idle timer without creating the resource. */
  touch(): void {
    this.clearTimer();
    if (this.config.idleTimeoutMs > 0 && this.resource !== null) {
      this.idleTimer = setTimeout(() => {
        if (this.config.canEvict && !this.config.canEvict()) {
          this.touch(); // reschedule
          return;
        }
        this.evict().catch(err => {
          console.error(`Error evicting idle ${this.config.name}:`, err);
        });
      }, this.config.idleTimeoutMs);
      this.idleTimer.unref();
    }
  }

  /** Evict (dispose) the resource if loaded. */
  async evict(): Promise<void> {
    this.clearTimer();
    if (this.resource === null) return;
    const r = this.resource;
    this.resource = null;
    await this.config.dispose(r);
  }

  /** Hard dispose — shutdown path. No canEvict check. Future get() throws. */
  async dispose(): Promise<void> {
    if (this._disposed) return;
    this._disposed = true;
    this.clearTimer();
    if (this.createPromise) {
      try { await this.createPromise; } catch { /* creation may throw */ }
    }
    if (this.resource !== null) {
      const r = this.resource;
      this.resource = null;
      await this.config.dispose(r);
    }
  }

  private clearTimer(): void {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
  }
}

// =============================================================================
// node-llama-cpp Implementation
// =============================================================================

export type LlamaCppConfig = {
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  modelCacheDir?: string;
  /**
   * Inactivity timeout in ms before unloading contexts (default: 2 minutes, 0 to disable).
   *
   * Per node-llama-cpp lifecycle guidance, we prefer keeping models loaded and only disposing
   * contexts when idle, since contexts (and their sequences) are the heavy per-session objects.
   * @see https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
   */
  inactivityTimeoutMs?: number;
  /**
   * Whether to dispose models on inactivity (default: false).
   *
   * Keeping models loaded avoids repeated VRAM thrash; set to true only if you need aggressive
   * memory reclaim.
   */
  disposeModelsOnInactivity?: boolean;
};

/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 5 minutes (keep models warm during typical search sessions)
const DEFAULT_INACTIVITY_TIMEOUT_MS = 5 * 60 * 1000;

export class LlamaCpp implements LLM {
  // IdleResourceSlots (lazy-create + per-slot idle-evict)
  private llamaSlot: IdleResourceSlot<Llama>;
  private embedModelSlot: IdleResourceSlot<LlamaModel>;
  private embedContextsSlot: IdleResourceSlot<LlamaEmbeddingContext[]>;
  private generateModelSlot: IdleResourceSlot<LlamaModel>;
  private rerankModelSlot: IdleResourceSlot<LlamaModel>;
  private rerankContextsSlot: IdleResourceSlot<Awaited<ReturnType<LlamaModel["createRankingContext"]>>[]>;

  // Simple nullable (trivial cost, no eviction needed)
  private expandGrammar: LlamaGrammar | null = null;

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;

  // Track disposal state to prevent double-dispose
  private disposed = false;

  // Qwen3 reranker template adds ~200 tokens overhead (system prompt, tags, etc.)
  // Chunks are max 800 tokens, so 800 + 200 + query ≈ 1100 tokens typical.
  // Use 2048 for safety margin. Still 17× less than auto (40960).
  private static readonly RERANK_CONTEXT_SIZE = 2048;

  constructor(config: LlamaCppConfig = {}) {
    this.embedModelUri = config.embedModel || DEFAULT_EMBED_MODEL;
    this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;

    const inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
    const disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
    const modelTimeoutMs = disposeModelsOnInactivity ? inactivityTimeoutMs : 0;

    // --- Llama instance (lightweight, never auto-evict) ---
    this.llamaSlot = new IdleResourceSlot({
      name: "llama",
      create: () => this.initLlama(),
      dispose: async (llama) => {
        // llama.dispose() can hang indefinitely, so use a timeout
        const p = llama.dispose();
        const timeout = new Promise<void>(r => setTimeout(r, 2000));
        await Promise.race([p, timeout]);
      },
      idleTimeoutMs: 0,
    });

    // --- Embed model ---
    this.embedModelSlot = new IdleResourceSlot({
      name: "embed-model",
      create: async () => {
        const llama = await this.llamaSlot.get();
        const modelPath = await this.resolveModel(this.embedModelUri);
        return llama.loadModel({ modelPath });
      },
      dispose: async (model) => {
        await this.embedContextsSlot.evict();
        await model.dispose();
      },
      idleTimeoutMs: modelTimeoutMs,
      canEvict: () => canUnloadLLM() && !this.embedContextsSlot.isLoaded,
    });

    // --- Embed contexts (pool for parallel embedding) ---
    this.embedContextsSlot = new IdleResourceSlot({
      name: "embed-contexts",
      create: async () => {
        const model = await this.embedModelSlot.get();
        // Embed contexts are ~143 MB each (nomic-embed 2048 ctx)
        const n = await this.computeParallelism(150);
        const threads = await this.threadsPerContext(n);
        const contexts: LlamaEmbeddingContext[] = [];
        for (let i = 0; i < n; i++) {
          try {
            contexts.push(await model.createEmbeddingContext({
              ...(threads > 0 ? { threads } : {}),
            }));
          } catch {
            if (contexts.length === 0) throw new Error("Failed to create any embedding context");
            break;
          }
        }
        return contexts;
      },
      dispose: async (contexts) => {
        for (const ctx of contexts) {
          try { await ctx.dispose(); } catch { /* already disposed */ }
        }
      },
      idleTimeoutMs: inactivityTimeoutMs,
      canEvict: () => canUnloadLLM(),
    });

    // --- Generate model ---
    this.generateModelSlot = new IdleResourceSlot({
      name: "generate-model",
      create: async () => {
        const llama = await this.llamaSlot.get();
        const modelPath = await this.resolveModel(this.generateModelUri);
        return llama.loadModel({ modelPath });
      },
      dispose: async (model) => {
        this.expandGrammar = null;
        await model.dispose();
      },
      idleTimeoutMs: modelTimeoutMs,
      canEvict: () => canUnloadLLM(),
    });

    // --- Rerank model ---
    this.rerankModelSlot = new IdleResourceSlot({
      name: "rerank-model",
      create: async () => {
        const llama = await this.llamaSlot.get();
        const modelPath = await this.resolveModel(this.rerankModelUri);
        return llama.loadModel({ modelPath });
      },
      dispose: async (model) => {
        await this.rerankContextsSlot.evict();
        await model.dispose();
      },
      idleTimeoutMs: modelTimeoutMs,
      canEvict: () => canUnloadLLM() && !this.rerankContextsSlot.isLoaded,
    });

    // --- Rerank contexts (pool for parallel ranking) ---
    this.rerankContextsSlot = new IdleResourceSlot({
      name: "rerank-contexts",
      create: async () => {
        const model = await this.rerankModelSlot.get();
        // ~960 MB per context with flash attention at contextSize 2048
        const n = await this.computeParallelism(1000);
        const threads = await this.threadsPerContext(n);
        const contexts: Awaited<ReturnType<LlamaModel["createRankingContext"]>>[] = [];
        for (let i = 0; i < n; i++) {
          try {
            contexts.push(await model.createRankingContext({
              contextSize: LlamaCpp.RERANK_CONTEXT_SIZE,
              flashAttention: true,
              ...(threads > 0 ? { threads } : {}),
            }));
          } catch {
            if (contexts.length === 0) {
              // Flash attention might not be supported — retry without it
              try {
                contexts.push(await model.createRankingContext({
                  contextSize: LlamaCpp.RERANK_CONTEXT_SIZE,
                  ...(threads > 0 ? { threads } : {}),
                }));
              } catch {
                throw new Error("Failed to create any rerank context");
              }
            }
            break;
          }
        }
        return contexts;
      },
      dispose: async (contexts) => {
        for (const ctx of contexts) {
          try { await ctx.dispose(); } catch { /* already disposed */ }
        }
      },
      idleTimeoutMs: inactivityTimeoutMs,
      canEvict: () => canUnloadLLM(),
    });
  }

  /**
   * Ensure model cache directory exists
   */
  private ensureModelCacheDir(): void {
    if (!existsSync(this.modelCacheDir)) {
      mkdirSync(this.modelCacheDir, { recursive: true });
    }
  }

  /**
   * Create and return a new llama instance (called by llamaSlot.create).
   */
  private async initLlama(): Promise<Llama> {
    const VALID_DEVICES = ["cpu", "cuda", "metal", "vulkan"] as const;
    type Device = (typeof VALID_DEVICES)[number];
    const deviceEnv = process.env.QMD_DEVICE?.toLowerCase().trim();

    let llama: Llama;

    if (deviceEnv) {
      // Explicit device override via QMD_DEVICE
      if (!VALID_DEVICES.includes(deviceEnv as Device)) {
        throw new Error(
          `QMD_DEVICE="${process.env.QMD_DEVICE}" is not valid. Valid options: ${VALID_DEVICES.join(", ")}`
        );
      }
      const gpu = deviceEnv === "cpu" ? false : (deviceEnv as "cuda" | "metal" | "vulkan");
      llama = await getLlama({ gpu, logLevel: LlamaLogLevel.error });
    } else {
      // Auto-detect: prefer CUDA > Metal > Vulkan > CPU.
      // We can't rely on gpu:"auto" — it returns false even when CUDA is available
      // (likely a binary/build config issue in node-llama-cpp).
      const gpuTypes = await getLlamaGpuTypes();
      const preferred = (["cuda", "metal", "vulkan"] as const).find(g => gpuTypes.includes(g));

      if (preferred) {
        try {
          llama = await getLlama({ gpu: preferred, logLevel: LlamaLogLevel.error });
        } catch {
          llama = await getLlama({ gpu: false, logLevel: LlamaLogLevel.error });
          process.stderr.write(
            `QMD Warning: ${preferred} reported available but failed to initialize. Falling back to CPU.\n`
          );
        }
      } else {
        llama = await getLlama({ gpu: false, logLevel: LlamaLogLevel.error });
      }

      if (!llama.gpu) {
        process.stderr.write(
          "QMD Warning: no GPU acceleration, running on CPU (slow). Run 'qmd status' for details.\n"
        );
      }
    }

    return llama;
  }

  /**
   * Resolve a model URI to a local path, downloading if needed
   */
  private async resolveModel(modelUri: string): Promise<string> {
    this.ensureModelCacheDir();
    // resolveModelFile handles HF URIs and downloads to the cache dir
    return await resolveModelFile(modelUri, this.modelCacheDir);
  }

  /**
   * Compute how many parallel contexts to create.
   *
   * GPU: constrained by VRAM (25% of free, capped at 8).
   * CPU: constrained by cores. Splitting threads across contexts enables
   *      true parallelism (each context runs on its own cores). Use at most
   *      half the math cores, with at least 4 threads per context.
   */
  private async computeParallelism(perContextMB: number): Promise<number> {
    const llama = await this.llamaSlot.get();

    const cap = parseInt(process.env.QMD_MAX_CONTEXTS || "0", 10);

    // GPU: one context per device
    if (llama.gpu) {
      const gpuDevices = await llama.getGpuDeviceNames();
      const n = Math.max(1, gpuDevices.length);
      return cap > 0 ? Math.min(n, cap) : n;
    }

    // CPU: split cores across contexts. At least 4 threads per context.
    const cores = llama.cpuMathCores || 4;
    const maxContexts = Math.floor(cores / 4);
    const n = Math.max(1, Math.min(4, maxContexts));
    return cap > 0 ? Math.min(n, cap) : n;
  }

  /**
   * Get the number of threads each context should use, given N parallel contexts.
   * Splits available math cores evenly across contexts.
   */
  private async threadsPerContext(parallelism: number): Promise<number> {
    const llama = await this.llamaSlot.get();
    if (llama.gpu) return 0; // GPU: let the library decide
    const cores = llama.cpuMathCores || 4;
    return Math.max(1, Math.floor(cores / parallelism));
  }

  /**
   * Get or create the cached expand grammar (GBNF compiled once).
   */
  private async ensureExpandGrammar(): Promise<LlamaGrammar> {
    if (!this.expandGrammar) {
      const llama = await this.llamaSlot.get();
      this.expandGrammar = await llama.createGrammar({
        grammar: `
          root ::= line+
          line ::= type ": " content "\\n"
          type ::= "lex" | "vec" | "hyde"
          content ::= [^\\n]+
        `
      });
    }
    return this.expandGrammar;
  }


  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the embedding model's tokenizer
   * Returns tokenizer tokens (opaque type from node-llama-cpp)
   */
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
    const model = await this.embedModelSlot.get();
    return model.tokenize(text);
  }

  /**
   * Count tokens in text using the embedding model's tokenizer
   */
  async countTokens(text: string): Promise<number> {
    return (await this.tokenize(text)).length;
  }

  /**
   * Detokenize token IDs back to text
   */
  async detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    const model = await this.embedModelSlot.get();
    return model.detokenize(tokens);
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const context = (await this.embedContextsSlot.get())[0]!;
      const embedding = await context.getEmbeddingFor(text);
      return {
        embedding: Array.from(embedding.vector),
        model: this.embedModelUri,
      };
    } catch (error) {
      console.error("Embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      const contexts = await this.embedContextsSlot.get();
      const n = contexts.length;

      if (n === 1) {
        // Single context: sequential (no point splitting)
        const context = contexts[0]!;
        const embeddings = [];
        for (const text of texts) {
          try {
            const embedding = await context.getEmbeddingFor(text);
            this.embedContextsSlot.touch();
            embeddings.push({ embedding: Array.from(embedding.vector), model: this.embedModelUri });
          } catch (err) {
            console.error("Embedding error for text:", err);
            embeddings.push(null);
          }
        }
        return embeddings;
      }

      // Multiple contexts: split texts across contexts for parallel evaluation
      const chunkSize = Math.ceil(texts.length / n);
      const chunks = Array.from({ length: n }, (_, i) =>
        texts.slice(i * chunkSize, (i + 1) * chunkSize)
      );

      const chunkResults = await Promise.all(
        chunks.map(async (chunk, i) => {
          const ctx = contexts[i]!;
          const results: (EmbeddingResult | null)[] = [];
          for (const text of chunk) {
            try {
              const embedding = await ctx.getEmbeddingFor(text);
              this.embedContextsSlot.touch();
              results.push({ embedding: Array.from(embedding.vector), model: this.embedModelUri });
            } catch (err) {
              console.error("Embedding error for text:", err);
              results.push(null);
            }
          }
          return results;
        })
      );

      return chunkResults.flat();
    } catch (error) {
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const model = await this.generateModelSlot.get();

    // Create fresh context -> sequence -> session for each call
    const context = await model.createContext();
    const sequence = context.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence });

    const maxTokens = options.maxTokens ?? 150;
    // Qwen3 recommends temp=0.7, topP=0.8, topK=20 for non-thinking mode
    // DO NOT use greedy decoding (temp=0) - causes repetition loops
    const temperature = options.temperature ?? 0.7;

    let result = "";
    try {
      await session.prompt(prompt, {
        maxTokens,
        temperature,
        topK: 20,
        topP: 0.8,
        onTextChunk: (text) => {
          result += text;
        },
      });

      return {
        text: result,
        model: this.generateModelUri,
        done: true,
      };
    } finally {
      // Dispose context (which disposes dependent sequences/sessions per lifecycle rules)
      await context.dispose();
    }
  }

  async modelExists(modelUri: string): Promise<ModelInfo> {
    // For HuggingFace URIs, we assume they exist
    // For local paths, check if file exists
    if (modelUri.startsWith("hf:")) {
      return { name: modelUri, exists: true };
    }

    const exists = existsSync(modelUri);
    return {
      name: modelUri,
      exists,
      path: exists ? modelUri : undefined,
    };
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, options: { context?: string, includeLexical?: boolean } = {}): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;

    const grammar = await this.ensureExpandGrammar();
    // Fresh context per call — LlamaChatSession accumulates chat history into
    // the KV cache, so reusing a context across calls pollutes subsequent runs.
    const model = await this.generateModelSlot.get();
    const context = await model.createContext();
    const sequence = context.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence });

    const prompt = `/no_think Expand this search query: ${query}`;

    try {
      // Qwen3 recommended settings for non-thinking mode:
      // temp=0.7, topP=0.8, topK=20, presence_penalty for repetition
      // DO NOT use greedy decoding (temp=0) - causes infinite loops
      const result = await session.prompt(prompt, {
        grammar,
        maxTokens: 600,
        temperature: 0.7,
        topK: 20,
        topP: 0.8,
        repeatPenalty: {
          lastTokens: 64,
          presencePenalty: 0.5,
        },
      });

      const lines = result.trim().split("\n");
      const queryLower = query.toLowerCase();
      const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

      const hasQueryTerm = (text: string): boolean => {
        const lower = text.toLowerCase();
        if (queryTerms.length === 0) return true;
        return queryTerms.some(term => lower.includes(term));
      };

      const queryables: Queryable[] = lines.map(line => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!hasQueryTerm(text)) return null;
        return { type: type as QueryType, text };
      }).filter((q): q is Queryable => q !== null);

      // Filter out lex entries if not requested
      const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
      if (filtered.length > 0) return filtered;

      const fallback: Queryable[] = [
        { type: 'hyde', text: `Information about ${query}` },
        { type: 'lex', text: query },
        { type: 'vec', text: query },
      ];
      return includeLexical ? fallback : fallback.filter(q => q.type !== 'lex');
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    } finally {
      // Dispose context (which disposes dependent sequences per lifecycle rules)
      await context.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    const contexts = await this.rerankContextsSlot.get();

    // Build a map from document text to original indices (for lookup after sorting)
    const textToDoc = new Map<string, { file: string; index: number }>();
    documents.forEach((doc, index) => {
      textToDoc.set(doc.text, { file: doc.file, index });
    });

    // Extract just the text for ranking
    const texts = documents.map((doc) => doc.text);

    // Split documents across contexts for parallel evaluation.
    // Each context has its own sequence with a lock, so parallelism comes
    // from multiple contexts evaluating different chunks simultaneously.
    const n = contexts.length;
    const chunkSize = Math.ceil(texts.length / n);
    const chunks = Array.from({ length: n }, (_, i) =>
      texts.slice(i * chunkSize, (i + 1) * chunkSize)
    ).filter(chunk => chunk.length > 0);

    const allScores = await Promise.all(
      chunks.map((chunk, i) => contexts[i]!.rankAll(query, chunk))
    );

    // Reassemble scores in original order and sort
    const flatScores = allScores.flat();
    const ranked = texts
      .map((text, i) => ({ document: text, score: flatScores[i]! }))
      .sort((a, b) => b.score - a.score);

    // Map back to our result format using the text-to-doc map
    const results: RerankDocumentResult[] = ranked.map((item) => {
      const docInfo = textToDoc.get(item.document)!;
      return {
        file: docInfo.file,
        score: item.score,
        index: docInfo.index,
      };
    });

    return {
      results,
      model: this.rerankModelUri,
    };
  }

  /**
   * Get device/GPU info for status display.
   * Initializes llama if not already done.
   */
  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    const llama = await this.llamaSlot.get();
    const gpuDevices = await llama.getGpuDeviceNames();
    let vram: { total: number; used: number; free: number } | undefined;
    if (llama.gpu) {
      try {
        const state = await llama.getVramState();
        vram = { total: state.total, used: state.used, free: state.free };
      } catch { /* no vram info */ }
    }
    return {
      gpu: llama.gpu,
      gpuOffloading: llama.supportsGpuOffloading,
      gpuDevices,
      vram,
      cpuCores: llama.cpuMathCores,
    };
  }

  /**
   * Get current context counts (for diagnostics/benchmarking).
   * Returns 0 for contexts that haven't been lazily created yet.
   */
  getContextCounts(): { embed: number; rerank: number } {
    return {
      embed: this.embedContextsSlot.current?.length ?? 0,
      rerank: this.rerankContextsSlot.current?.length ?? 0,
    };
  }

  /**
   * Force-evict idle resources (contexts, optionally models) without waiting
   * for timers. Useful for tests and manual VRAM management. Resources reload
   * transparently on next use.
   */
  async evictIdle(): Promise<void> {
    if (this.disposed) return;
    await this.embedContextsSlot.evict();
    await this.rerankContextsSlot.evict();
    this.expandGrammar = null;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;

    // Dispose in reverse order of creation: contexts → models → llama.
    // This follows node-llama-cpp's required disposal hierarchy to avoid
    // dangling native references and NAPI crashes on exit.
    // See: https://node-llama-cpp.withcat.ai/guide/objects-lifecycle

    // 1. Context pools first (depend on models)
    await this.embedContextsSlot.dispose();
    await this.rerankContextsSlot.dispose();
    this.expandGrammar = null;

    // 2. Models next (depend on llama)
    await this.embedModelSlot.dispose();
    await this.generateModelSlot.dispose();
    await this.rerankModelSlot.dispose();

    // 3. Llama last (owns the GPU context)
    await this.llamaSlot.dispose();
  }
}

// =============================================================================
// Session Management Layer
// =============================================================================

/**
 * Manages LLM session lifecycle with reference counting.
 * Coordinates with LlamaCpp idle timeout to prevent disposal during active sessions.
 */
class LLMSessionManager {
  private llm: LlamaCpp;
  private _activeSessionCount = 0;
  private _inFlightOperations = 0;

  constructor(llm: LlamaCpp) {
    this.llm = llm;
  }

  get activeSessionCount(): number {
    return this._activeSessionCount;
  }

  get inFlightOperations(): number {
    return this._inFlightOperations;
  }

  /**
   * Returns true only when both session count and in-flight operations are 0.
   * Used by LlamaCpp to determine if idle unload is safe.
   */
  canUnload(): boolean {
    return this._activeSessionCount === 0 && this._inFlightOperations === 0;
  }

  acquire(): void {
    this._activeSessionCount++;
  }

  release(): void {
    this._activeSessionCount = Math.max(0, this._activeSessionCount - 1);
  }

  operationStart(): void {
    this._inFlightOperations++;
  }

  operationEnd(): void {
    this._inFlightOperations = Math.max(0, this._inFlightOperations - 1);
  }

  getLlamaCpp(): LlamaCpp {
    return this.llm;
  }
}

/**
 * Error thrown when an operation is attempted on a released or aborted session.
 */
export class SessionReleasedError extends Error {
  constructor(message = "LLM session has been released or aborted") {
    super(message);
    this.name = "SessionReleasedError";
  }
}

/**
 * Scoped LLM session with automatic lifecycle management.
 * Wraps LlamaCpp methods with operation tracking and abort handling.
 */
class LLMSession implements ILLMSession {
  private manager: LLMSessionManager;
  private released = false;
  private abortController: AbortController;
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
  private name: string;

  constructor(manager: LLMSessionManager, options: LLMSessionOptions = {}) {
    this.manager = manager;
    this.name = options.name || "unnamed";
    this.abortController = new AbortController();

    // Link external abort signal if provided
    if (options.signal) {
      if (options.signal.aborted) {
        this.abortController.abort(options.signal.reason);
      } else {
        options.signal.addEventListener("abort", () => {
          this.abortController.abort(options.signal!.reason);
        }, { once: true });
      }
    }

    // Set up max duration timer
    const maxDuration = options.maxDuration ?? 10 * 60 * 1000; // Default 10 minutes
    if (maxDuration > 0) {
      this.maxDurationTimer = setTimeout(() => {
        this.abortController.abort(new Error(`Session "${this.name}" exceeded max duration of ${maxDuration}ms`));
      }, maxDuration);
      this.maxDurationTimer.unref(); // Don't keep process alive
    }

    // Acquire session lease
    this.manager.acquire();
  }

  get isValid(): boolean {
    return !this.released && !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  /**
   * Release the session and decrement ref count.
   * Called automatically by withLLMSession when the callback completes.
   */
  release(): void {
    if (this.released) return;
    this.released = true;

    if (this.maxDurationTimer) {
      clearTimeout(this.maxDurationTimer);
      this.maxDurationTimer = null;
    }

    this.abortController.abort(new Error("Session released"));
    this.manager.release();
  }

  /**
   * Wrap an operation with tracking and abort checking.
   */
  private async withOperation<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.isValid) {
      throw new SessionReleasedError();
    }

    this.manager.operationStart();
    try {
      // Check abort before starting
      if (this.abortController.signal.aborted) {
        throw new SessionReleasedError(
          this.abortController.signal.reason?.message || "Session aborted"
        );
      }
      return await fn();
    } finally {
      this.manager.operationEnd();
    }
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.withOperation(() => this.manager.getLlamaCpp().embed(text, options));
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.withOperation(() => this.manager.getLlamaCpp().embedBatch(texts));
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    return this.withOperation(() => this.manager.getLlamaCpp().expandQuery(query, options));
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    return this.withOperation(() => this.manager.getLlamaCpp().rerank(query, documents, options));
  }
}

// Session manager for the default LlamaCpp instance
let defaultSessionManager: LLMSessionManager | null = null;

/**
 * Get the session manager for the default LlamaCpp instance.
 */
function getSessionManager(): LLMSessionManager {
  const llm = getDefaultLlamaCpp();
  if (!defaultSessionManager || defaultSessionManager.getLlamaCpp() !== llm) {
    defaultSessionManager = new LLMSessionManager(llm);
  }
  return defaultSessionManager;
}

/**
 * Execute a function with a scoped LLM session.
 * The session provides lifecycle guarantees - resources won't be disposed mid-operation.
 *
 * @example
 * ```typescript
 * await withLLMSession(async (session) => {
 *   const expanded = await session.expandQuery(query);
 *   const embeddings = await session.embedBatch(texts);
 *   const reranked = await session.rerank(query, docs);
 *   return reranked;
 * }, { maxDuration: 10 * 60 * 1000, name: 'querySearch' });
 * ```
 */
export async function withLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const manager = getSessionManager();
  const session = new LLMSession(manager, options);

  try {
    return await fn(session);
  } finally {
    session.release();
  }
}

/**
 * Check if idle unload is safe (no active sessions or operations).
 * Used internally by LlamaCpp idle timer.
 */
export function canUnloadLLM(): boolean {
  if (!defaultSessionManager) return true;
  return defaultSessionManager.canUnload();
}

// =============================================================================
// Singleton for default LlamaCpp instance
// =============================================================================

let defaultLlamaCpp: LlamaCpp | null = null;

/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export function getDefaultLlamaCpp(): LlamaCpp {
  if (!defaultLlamaCpp) {
    defaultLlamaCpp = new LlamaCpp();
  }
  return defaultLlamaCpp;
}

/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm: LlamaCpp | null): void {
  defaultLlamaCpp = llm;
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  if (defaultLlamaCpp) {
    await defaultLlamaCpp.dispose();
    defaultLlamaCpp = null;
  }
}
