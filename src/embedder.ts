import { join, basename } from "node:path";
import { existsSync } from "node:fs";
import { homedir } from "node:os";

/**
 * Dual ONNX embedding pipelines via Transformers.js, mirroring
 * pi-local-rag's two embedding groups:
 *
 * - text  → nomic-ai/nomic-embed-text-v1.5 (prose, markup, data/config)
 * - code  → jinaai/jina-embeddings-v2-base-code (source code)
 *
 * Both are 768-dim and run locally (q8 quantized ONNX) — no embedding
 * APIs. Model downloads are cached in a shared HuggingFace cache
 * directory so they happen once per machine.
 */

/** Embedding group: code files go to the code model, everything else to the text model. */
export type EmbedGroup = "code" | "text";

/** HuggingFace id of the text/prose embedding model. */
export const EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5";

/** nomic's fixed embedding dimensionality (no matryoshka truncation). */
export const EMBEDDING_DIMENSIONS = 768;

/**
 * HuggingFace id of the code embedding model — the official repo ships
 * transformers.js-compatible ONNX weights, including the quantized (q8)
 * file. 768-dim, trained on code + docstring pairs.
 */
export const CODE_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code";

/** Both models share nomic's 768-dim space size. */
export const CODE_EMBEDDING_DIMENSIONS = 768;

/** HuggingFace id of the embedding model for one group. */
export function embeddingModelFor(group: EmbedGroup): string {
  return group === "code" ? CODE_EMBEDDING_MODEL : EMBEDDING_MODEL;
}

/**
 * nomic-embed-text-v1.5 asymmetric-retrieval task prefixes (see model
 * card): queries get `search_query:`, documents `search_document:`.
 */
const TRANSFORMERS_QUERY_PREFIX = "search_query: ";
const TRANSFORMERS_DOC_PREFIX = "search_document: ";

/**
 * jina v2 models take NO task prefixes (those arrived with v3) — its
 * code-search training maps natural language straight onto code.
 */
const CODE_QUERY_PREFIX = "";
const CODE_DOC_PREFIX = "";

/**
 * Unified embedding interface. The only implementation is the local
 * Transformers.js (ONNX) engine — this extension is fully local, no remote
 * embedding APIs. `group` selects the model (default: text/nomic).
 */
export interface Embedder {
  embed(text: string, group?: EmbedGroup, signal?: AbortSignal): Promise<number[]>;
  embedBatch(
    texts: string[],
    group?: EmbedGroup,
    signal?: AbortSignal,
    concurrency?: number
  ): Promise<(number[] | null)[]>;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createEmbedder(): Embedder {
  return new TransformersEmbedder();
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

/**
 * The exact input text a model expects for a QUERY. nomic uses its
 * `search_query:` task prefix; jina v2 takes the raw question as-is.
 */
export function buildQueryInput(group: EmbedGroup, text: string): string {
  const prefix = group === "code" ? CODE_QUERY_PREFIX : TRANSFORMERS_QUERY_PREFIX;
  return prefix + text.replace(/\s+/g, " ").trim();
}

/**
 * The exact input text a model expects for a DOCUMENT chunk. nomic uses
 * `search_document:`; jina v2 takes no prefix, but the file basename is
 * prepended as a context line (pi-local-rag's file-context scheme): jina-code
 * was trained on code-with-context pairs, and a bare slice loses its file
 * identity otherwise — the basename anchors filename-oriented queries
 * ("what does auth.ts do?") without touching the stored chunk content.
 */
export function buildDocumentInput(group: EmbedGroup, text: string, filename?: string): string {
  const prefix = group === "code" ? CODE_DOC_PREFIX : TRANSFORMERS_DOC_PREFIX;
  const content = group === "code" && filename ? `${basename(filename)}\n${text}` : text;
  return prefix + content;
}

/** Texts per single ONNX forward pass — bounds padded-batch wall time on CPU. */
const TRANSFORMERS_BATCH_SIZE = 16;

/**
 * Persistent HuggingFace model-cache directory, shared with pi-local-rag so
 * the ~111 MB nomic + ~170 MB jina-code downloads happen once per machine.
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
 * explain a cold-start download (~111 MB nomic, ~170 MB jina-code) before
 * indexing appears to stall, mirroring pi-local-rag's onModelLoad notice.
 */
export function isTransformersModelCached(model: string): boolean {
  return existsSync(join(resolveTransformersCacheDir(), model, "onnx", "model_quantized.onnx"));
}

class TransformersEmbedder implements Embedder {
  /** One pipeline load promise per group so both models coexist lazily. */
  private pipelinePromises = new Map<EmbedGroup, Promise<unknown> | null>();

  /**
   * Lazily load the ONNX feature-extraction pipeline (q8 quantized weights)
   * for a group. The load promise is cached so concurrent first calls share
   * a single download; a failed load is evicted so the next call retries.
   */
  private getPipeline(group: EmbedGroup): Promise<any> {
    const existing = this.pipelinePromises.get(group);
    if (existing) return existing;
    const loadPromise = (async () => {
      const { pipeline, env } = await import("@huggingface/transformers");
      env.cacheDir = resolveTransformersCacheDir();
      return pipeline("feature-extraction", embeddingModelFor(group), { dtype: "q8" });
    })();
    this.pipelinePromises.set(group, loadPromise);
    loadPromise.catch(() => {
      this.pipelinePromises.set(group, null);
    });
    return loadPromise;
  }

  async embed(text: string, group: EmbedGroup = "text", signal?: AbortSignal): Promise<number[]> {
    if (signal?.aborted) throw new Error("Aborted");
    const pipe = await this.getPipeline(group);
    const output = await pipe(truncate(buildQueryInput(group, text)), {
      pooling: "mean",
      normalize: true,
    });
    return Array.from(output.data as Float32Array);
  }

  async embedBatch(
    texts: string[],
    group: EmbedGroup = "text",
    signal?: AbortSignal,
    _concurrency?: number
  ): Promise<(number[] | null)[]> {
    const results: (number[] | null)[] = new Array(texts.length).fill(null);
    if (texts.length === 0) return results;

    let failed = 0;
    const errs = new Set<string>();
    try {
      const pipe = await this.getPipeline(group);
      for (let start = 0; start < texts.length; start += TRANSFORMERS_BATCH_SIZE) {
        if (signal?.aborted) throw new Error("Aborted");
        const batch = texts
          .slice(start, start + TRANSFORMERS_BATCH_SIZE)
          .map((t) => truncate(t));
        // One forward pass per batch — the pooled output Tensor has dims
        // [batchSize, dim]; sliced per-text. Documents get the group's doc
        // prefix (nomic task instruction; jina v2 uses none). The filename
        // context for code chunks is part of the stored text (built by the
        // chunk-embed-text step), so no per-text filename is needed here.
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
          `Transformers embedding failed for ${failed}/${texts.length} chunks (${embeddingModelFor(group)}): ${summarizeErrors(errs)}`
        );
      }
    }
    return results;
  }
}
