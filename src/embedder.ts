import { join } from "node:path";
import { homedir } from "node:os";
import type { ProviderConfig } from "./config.js";

/**
 * Unified embedding interface. Implementations for OpenAI, OpenAI-compatible,
 * Bedrock, Ollama, and local Transformers.js (ONNX).
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

export function createEmbedder(config: ProviderConfig, dimensions: number): Embedder {
  switch (config.type) {
    case "openai":
      return new OpenAIEmbedder(config.apiKey, config.model, dimensions, undefined);
    case "openai-compatible":
      return new OpenAIEmbedder(config.apiKey ?? "", config.model, dimensions, config.baseUrl);
    case "bedrock":
      return new BedrockEmbedder(config.profile, config.region, config.model, dimensions);
    case "ollama":
      return new OllamaEmbedder(config.url, config.model);
    case "transformers":
      return new TransformersEmbedder(config.model);
  }
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

const RETRY_DELAYS = [1000, 2000, 4000]; // exponential backoff for 429s

/** Retry a fetch-based operation on 429 rate-limit errors with exponential backoff. */
async function withRateLimitRetry<T>(fn: () => Promise<T>, label: string): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (err: any) {
      const is429 =
        err?.message?.includes("429") ||
        err?.name === "ThrottlingException" ||
        err?.$metadata?.httpStatusCode === 429;
      if (is429 && attempt < RETRY_DELAYS.length) {
        const delay = RETRY_DELAYS[attempt];
        console.error(
          `knowledge-search: ${label} rate limited, retrying in ${delay}ms (attempt ${attempt + 1}/${RETRY_DELAYS.length})`
        );
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw err;
    }
  }
}

/** Run an async function over an array with bounded concurrency. */
async function parallelMap<T, R>(
  items: T[],
  fn: (item: T, index: number) => Promise<R>,
  concurrency: number,
  signal?: AbortSignal
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;

  const worker = async () => {
    while (cursor < items.length) {
      if (signal?.aborted) throw new Error("Aborted");
      const idx = cursor++;
      results[idx] = await fn(items[idx], idx);
    }
  };

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()));
  return results;
}

// ---------------------------------------------------------------------------
// OpenAI / OpenAI-compatible
// ---------------------------------------------------------------------------

class OpenAIEmbedder implements Embedder {
  private apiKey: string;
  private model: string;
  private dimensions: number;
  private endpoint: string;

  constructor(apiKey: string, model: string, dimensions: number, baseUrl?: string) {
    this.apiKey = apiKey;
    this.model = model;
    this.dimensions = dimensions;
    if (baseUrl) {
      this.endpoint = `${baseUrl.replace(/\/$/, "")}/v1/embeddings`;
    } else {
      this.endpoint = `https://api.openai.com/v1/embeddings`;
    }
  }

  async embed(text: string, signal?: AbortSignal): Promise<number[]> {
    const results = await this.embedBatch([text], signal);
    if (!results[0]) throw new Error("Embedding failed — provider returned no vector");
    return results[0];
  }

  async embedBatch(texts: string[], signal?: AbortSignal): Promise<(number[] | null)[]> {
    // OpenAI supports batch embedding natively (up to 2048 inputs).
    // Chunk into groups of 100 to stay safe on payload size.
    const BATCH = 100;
    const results: (number[] | null)[] = new Array(texts.length);

    for (let i = 0; i < texts.length; i += BATCH) {
      if (signal?.aborted) throw new Error("Aborted");
      const batch = texts.slice(i, i + BATCH).map((t) => truncate(t));

      try {
        const json = await withRateLimitRetry(async () => {
          const res = await fetch(this.endpoint, {
            method: "POST",
            headers: {
              ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              input: batch,
              model: this.model,
              dimensions: this.dimensions,
            }),
            signal,
          });

          if (!res.ok) {
            const body = await res.text();
            throw new Error(`OpenAI API ${res.status}: ${body.slice(0, 200)}`);
          }

          return (await res.json()) as {
            data: { embedding: number[]; index: number }[];
          };
        }, "embedding");

        for (const item of json.data) {
          results[i + item.index] = item.embedding;
        }
      } catch (err: any) {
        // Mark the whole batch as failed
        for (let j = 0; j < batch.length; j++) {
          results[i + j] = null;
        }
        const label = this.endpoint.includes("api.openai.com")
          ? "OpenAI"
          : `Embedding (${this.endpoint})`;
        console.error(`${label} batch embedding failed: ${err.message}`);
      }
    }

    return results;
  }
}

// ---------------------------------------------------------------------------
// Bedrock (Titan)
// ---------------------------------------------------------------------------

class BedrockEmbedder implements Embedder {
  private client: any; // Lazy-loaded to avoid hard dep if not using Bedrock
  private model: string;
  private dimensions: number;
  private clientPromise: Promise<any>;

  constructor(profile: string, region: string, model: string, dimensions: number) {
    this.model = model;
    this.dimensions = dimensions;

    // Lazy-load the AWS SDK — it's an optional dependency
    this.clientPromise = (async () => {
      const { BedrockRuntimeClient } = await import("@aws-sdk/client-bedrock-runtime");
      const { fromIni } = await import("@aws-sdk/credential-providers");
      return new BedrockRuntimeClient({
        region,
        credentials: fromIni({ profile }),
      });
    })();
  }

  async embed(text: string, signal?: AbortSignal): Promise<number[]> {
    const results = await this.embedBatch([text], signal);
    if (!results[0]) throw new Error("Embedding failed — provider returned no vector");
    return results[0];
  }

  async embedBatch(
    texts: string[],
    signal?: AbortSignal,
    concurrency = 10
  ): Promise<(number[] | null)[]> {
    const client = await this.clientPromise;

    let failed = 0;
    const errs = new Set<string>();
    const out = await parallelMap(
      texts,
      async (text) => {
        try {
          return await this.callBedrock(client, text);
        } catch (err: any) {
          failed++;
          errs.add(err.message);
          return null;
        }
      },
      concurrency,
      signal
    );
    // Aggregate to one line instead of one console.error per failed chunk —
    // a per-item log floods the TUI when the provider is unreachable. Report
    // the distinct error messages (capped) so a mix of failure modes is
    // still visible without re-introducing per-chunk spam.
    if (failed > 0) {
      console.error(
        `Bedrock embedding failed for ${failed}/${texts.length} chunks: ${summarizeErrors(errs)}`
      );
    }
    return out;
  }

  private async callBedrock(client: any, text: string): Promise<number[]> {
    return withRateLimitRetry(async () => {
      const { InvokeModelCommand } = await import("@aws-sdk/client-bedrock-runtime");

      const body = JSON.stringify({
        inputText: truncate(text),
        dimensions: this.dimensions,
        normalize: true,
      });

      const command = new InvokeModelCommand({
        modelId: this.model,
        contentType: "application/json",
        accept: "application/json",
        body: new TextEncoder().encode(body),
      });

      const response = await client.send(command);
      const responseBody = JSON.parse(new TextDecoder().decode(response.body));

      if (!responseBody.embedding) {
        throw new Error(
          "Unexpected Bedrock response: " + JSON.stringify(responseBody).slice(0, 200)
        );
      }
      return responseBody.embedding;
    }, "Bedrock embed");
  }
}

// ---------------------------------------------------------------------------
// Ollama
// ---------------------------------------------------------------------------

class OllamaEmbedder implements Embedder {
  private url: string;
  private model: string;

  constructor(url: string, model: string) {
    this.url = url.replace(/\/$/, "");
    this.model = model;
  }

  async embed(text: string, signal?: AbortSignal): Promise<number[]> {
    return withRateLimitRetry(async () => {
      const res = await fetch(`${this.url}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model, input: truncate(text) }),
        signal,
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Ollama API ${res.status}: ${body.slice(0, 200)}`);
      }

      const json = (await res.json()) as { embeddings: number[][] };
      return json.embeddings[0];
    }, "Ollama embed");
  }

  async embedBatch(
    texts: string[],
    signal?: AbortSignal,
    concurrency = 4
  ): Promise<(number[] | null)[]> {
    // Ollama /api/embed supports batch via `input` array
    // but some models/versions don't. Fall back to parallel single calls.
    let failed = 0;
    const errs = new Set<string>();
    const out = await parallelMap(
      texts,
      async (text) => {
        try {
          return await this.embed(text, signal);
        } catch (err: any) {
          failed++;
          errs.add(err.message);
          return null;
        }
      },
      concurrency,
      signal
    );
    // Aggregate to one line instead of one console.error per failed chunk —
    // a wedged Ollama would otherwise flood the TUI and corrupt the input box.
    // Report the distinct error messages (capped) so a mix of failure modes is
    // still visible without re-introducing per-chunk spam.
    if (failed > 0) {
      console.error(
        `Ollama embedding failed for ${failed}/${texts.length} chunks: ${summarizeErrors(errs)}`
      );
    }
    return out;
  }
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
