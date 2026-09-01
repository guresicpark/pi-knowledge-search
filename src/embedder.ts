import { join } from "node:path";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import type { ProviderConfig } from "./config.js";

/**
 * Unified embedding interface. The only implementation is the local
 * Transformers.js (ONNX) engine — this extension is fully local, no remote
 * embedding APIs.
 */
export interface Embedder {
  embed(text: string, signal?: AbortSignal): Promise<number[]>;
  embedBatch(
    texts: string[],
    signal?: AbortSignal,
    concurrency?: number
  ): Promise<(number[] | null)[]>;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createEmbedder(config: ProviderConfig, _dimensions: number): Embedder {
  return new TransformersEmbedder(config.model);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Truncate to stay within token limits. Conservative: ~10K chars ≈ 4-6K tokens. */
function truncate(text: string, maxChars = 10000): string {
  return text.length > maxChars ? text.slice(0, maxChars) : text;
}

/**
 * Summarize a set of distinct embedding-failure messages for a single log
 * line. Caps the number shown so a batch failing with many different errors
 * can't itself flood the output.
 */
function summarizeErrors(errs: Set<string>, max = 3): string {
  const list = [...errs];
  const shown = list.slice(0, max).join("; ");
  return list.length > max ? `${shown} (+${list.length - max} more)` : shown;
}

// ---------------------------------------------------------------------------
// Local Transformers.js (ONNX) — mirrors pi-local-rag's text-group pipeline
// ---------------------------------------------------------------------------

/** nomic-embed-text-v1.5 asymmetric-retrieval task prefixes (see model card). */
const TRANSFORMERS_QUERY_PREFIX = "search_query: ";
const TRANSFORMERS_DOC_PREFIX = "search_document: ";

/** Texts per single ONNX forward pass — bounds padded-batch wall time on CPU. */
const TRANSFORMERS_BATCH_SIZE = 16;

/**
 * Persistent HuggingFace model-cache directory, shared with pi-local-rag so
 * the ~111 MB nomic download happens once per machine.
 *
 * Priority: PI_RAG_MODEL_CACHE > TRANSFORMERS_CACHE > HF_HOME/transformers >
 * ~/.cache/huggingface/transformers.
 */
export function resolveTransformersCacheDir(): string {
  if (process.env.PI_RAG_MODEL_CACHE) return process.env.PI_RAG_MODEL_CACHE;
  if (process.env.TRANSFORMERS_CACHE) return process.env.TRANSFORMERS_CACHE;
  if (process.env.HF_HOME) return join(process.env.HF_HOME, "transformers");
  return join(homedir(), ".cache", "huggingface", "transformers");
}

/**
 * Whether a model's q8 ONNX weights are already present in the local
 * HuggingFace cache (Transformers.js stores them under
 * `<cacheDir>/<model>/onnx/model_quantized.onnx`). Callers use this to
 * explain a cold-start download (~111 MB for nomic) before indexing
 * appears to stall, mirroring pi-local-rag's onModelLoad notice.
 */
export function isTransformersModelCached(model: string): boolean {
  return existsSync(join(resolveTransformersCacheDir(), model, "onnx", "model_quantized.onnx"));
}

class TransformersEmbedder implements Embedder {
  private model: string;
  private pipelinePromise: Promise<unknown> | null = null;

  constructor(model: string) {
    this.model = model;
  }

  /**
   * Lazily load the ONNX feature-extraction pipeline (q8 quantized weights).
   * The load promise is cached so concurrent first calls share a single
   * download; a failed load is evicted so the next call retries.
   */
  private getPipeline(): Promise<any> {
    if (!this.pipelinePromise) {
      this.pipelinePromise = (async () => {
        const { pipeline, env } = await import("@huggingface/transformers");
        env.cacheDir = resolveTransformersCacheDir();
        return pipeline("feature-extraction", this.model, { dtype: "q8" });
      })();
      this.pipelinePromise.catch(() => {
        this.pipelinePromise = null;
      });
    }
    return this.pipelinePromise;
  }

  async embed(text: string, signal?: AbortSignal): Promise<number[]> {
    if (signal?.aborted) throw new Error("Aborted");
    const pipe = await this.getPipeline();
    // Query side: collapse whitespace (stray newlines dilute the embedding)
    // and apply the nomic search_query: task prefix.
    const input = TRANSFORMERS_QUERY_PREFIX + text.replace(/\s+/g, " ").trim();
    const output = await pipe(truncate(input), { pooling: "mean", normalize: true });
    return Array.from(output.data as Float32Array);
  }

  async embedBatch(
    texts: string[],
    signal?: AbortSignal,
    _concurrency?: number
  ): Promise<(number[] | null)[]> {
    const results: (number[] | null)[] = new Array(texts.length).fill(null);
    if (texts.length === 0) return results;

    let failed = 0;
    const errs = new Set<string>();
    try {
      const pipe = await this.getPipeline();
      for (let start = 0; start < texts.length; start += TRANSFORMERS_BATCH_SIZE) {
        if (signal?.aborted) throw new Error("Aborted");
        const batch = texts
          .slice(start, start + TRANSFORMERS_BATCH_SIZE)
          .map((t) => TRANSFORMERS_DOC_PREFIX + truncate(t));
        // One forward pass per batch — the pooled output Tensor has dims
        // [batchSize, dim]; sliced per-text.
        const output = await pipe(batch, { pooling: "mean", normalize: true });
        const flattened = output.data as Float32Array;
        const dim = flattened.length / batch.length;
        for (let i = 0; i < batch.length; i++) {
          results[start + i] = Array.from(flattened.slice(i * dim, (i + 1) * dim));
        }
      }
    } catch (err: any) {
      failed = results.filter((v) => v === null).length;
      errs.add(err.message);
      if (failed > 0) {
        console.error(
          `Transformers embedding failed for ${failed}/${texts.length} chunks: ${summarizeErrors(errs)}`
        );
      }
    }
    return results;
  }
}
