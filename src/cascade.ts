import { parseJevResponse } from './request.js';
import type { JevAnswer, JevAsker, JevQuestions, JevResponse, JevState } from './types.js';

/**
 * A cascade of decision models for compaction.
 *
 * A decision endpoint may serve several models and scale each to zero when
 * idle, so any of them may be cold. They are asked best first; a cold one is
 * skipped rather than waited on. When every hosted model is cold or failing,
 * an optional fallback model on an OpenAI-compatible server answers.
 */
export const DEFAULT_FALLBACK_MODEL = 'Qwen/Qwen3.5-4B';

/**
 * The longest state sent to the fallback model. Measured against a server
 * with an 8192-token context: 20,000 characters were answered and 30,000
 * refused with `400`.
 */
export const DEFAULT_FALLBACK_MAX_STATE_CHARS = 20_000;

/** How long one model gets before the cascade moves on; inside a 29 s gateway ceiling. */
export const PER_MODEL_TIMEOUT_MS = 12_000;

/**
 * What one request to a model may carry. A model may truncate a longer state
 * silently, so the state is checked here and the model skipped rather than
 * asked about a history it would only half read.
 */
export interface ModelLimits {
  maxQuestions: number;
  maxStateChars: number;
}

/** Limits for a model `CascadeOptions.limits` does not name. */
export const DEFAULT_MODEL_LIMITS: ModelLimits = { maxQuestions: 8, maxStateChars: 50_000 };

export type CascadeAuth = 'x-api-key' | 'bearer';

export type CascadeInit = {
  method: 'GET' | 'POST';
  headers: Record<string, string>;
  body?: string;
};

/** A fetch that has already read the body: `$.http.fetch` in the hook, `fetchTransport` elsewhere. */
export type CascadeTransport = (
  url: string,
  init: CascadeInit,
) => Promise<{ status: number; ok: boolean; text: string }>;

/** The fallback model: an OpenAI-compatible server and the alias it serves. */
export interface FallbackEndpoint {
  baseUrl: string;
  model: string;
  /** Longer states skip it without a request. Default `DEFAULT_FALLBACK_MAX_STATE_CHARS`. */
  maxStateChars?: number;
}

export interface CascadeOptions {
  /** Key for the hosted chain; without one the chain is skipped and only `fallback` is asked. */
  apiKey?: string;
  /** The decision endpoint; each model is asked at `{baseUrl}/{model}`. */
  baseUrl?: string;
  /** Models to try, best first. */
  models?: readonly string[];
  /** How the key is sent. Default `x-api-key`. */
  auth?: CascadeAuth;
  /** Per-model limits; a model not named gets `DEFAULT_MODEL_LIMITS`. */
  limits?: Readonly<Record<string, Partial<ModelLimits>>>;
  /** The last resort when every hosted model is skipped. */
  fallback?: FallbackEndpoint;
  transport: CascadeTransport;
  /** Per-call limit, enforced when `sleep` is given. Default `PER_MODEL_TIMEOUT_MS`. */
  timeoutMs?: number;
  /**
   * The wait the timeout races against: `$.clock.sleep` in the hook. Without
   * it the transport's own limit applies (`fetchTransport` sets one).
   */
  sleep?: (ms: number) => Promise<void>;
}

/** `a, b ,c` into `['a', 'b', 'c']`; blanks and repeats dropped. */
export function parseModelList(list: string): string[] {
  const models: string[] = [];
  for (const part of list.split(',')) {
    const model = part.trim();
    if (model && !models.includes(model)) models.push(model);
  }
  return models;
}

function trimBase(url: string): string {
  return url.replace(/\/+$/, '');
}

function authHeaders(apiKey: string, auth?: CascadeAuth): Record<string, string> {
  return auth === 'bearer' ? { authorization: `Bearer ${apiKey}` } : { 'x-api-key': apiKey };
}

/** The HTTP request for one model of the cascade. The model rides in the path, not the body. */
export function buildCascadeRequest(
  params: { apiKey: string; baseUrl: string; model: string; auth?: CascadeAuth },
  state: JevState,
  questions: JevQuestions,
): { url: string; init: CascadeInit } {
  return {
    url: `${trimBase(params.baseUrl)}/${encodeURIComponent(params.model)}`,
    init: {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...authHeaders(params.apiKey, params.auth) },
      body: JSON.stringify({ state, questions }),
    },
  };
}

/** Why a model's reply cannot be used, or `undefined` when every question has its `noul`. */
function unusable(response: JevResponse, questions: JevQuestions): string | undefined {
  for (const name of Object.keys(questions)) {
    const answer = response.answers[name];
    const noul = answer && 'noul' in answer ? answer.noul : undefined;
    if (typeof noul !== 'number' || !Number.isFinite(noul)) return `no answer for ${name}`;
  }
  return undefined;
}

function chunk<T>(items: readonly T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += size) chunks.push(items.slice(i, i + size));
  return chunks;
}

function stateText(state: JevState): string {
  return typeof state === 'string' ? state : JSON.stringify(state);
}

/** Everything outside the model's `<think>` blocks; an unterminated block means no answer arrived. */
export function stripThinking(text: string): string {
  let out = '';
  let rest = text;
  for (let open = rest.indexOf('<think>'); open >= 0; open = rest.indexOf('<think>')) {
    out += rest.slice(0, open);
    const close = rest.indexOf('</think>', open);
    if (close < 0) return out;
    rest = rest.slice(close + '</think>'.length);
  }
  return out + rest;
}

export type ChatChoice = {
  message?: { content?: string };
  logprobs?: {
    content?: { token?: string; top_logprobs?: { token?: string; logprob?: number }[] }[];
  };
};

/**
 * How much of the answer token's probability went to the option chosen, with
 * the mass outside the option markers dropped.
 * Zero when the reply carries no usable logprobs.
 */
export function markerConfidence(
  choice: ChatChoice,
  options: readonly string[],
  chosen: string,
): number {
  const generated = choice.logprobs?.content;
  if (!Array.isArray(generated)) return 0;
  const optionFor = (raw: string | undefined): string | undefined => {
    const token = (raw ?? '').trim().toLowerCase();
    if (token.length < 2) return undefined;
    return options.find((option) => option.toLowerCase().startsWith(token));
  };
  const answer = generated.find((token) => optionFor(token.token)) ?? generated[0];
  if (!answer) return 0;
  const mass = new Map<string, number>();
  for (const alternative of answer.top_logprobs ?? []) {
    const option = optionFor(alternative.token);
    if (!option || typeof alternative.logprob !== 'number') continue;
    mass.set(option, (mass.get(option) ?? 0) + Math.exp(alternative.logprob));
  }
  let total = 0;
  for (const value of mass.values()) total += value;
  return total > 0 ? (mass.get(chosen) ?? 0) / total : 0;
}

/**
 * Asks a chain of decision models and returns the first complete answer.
 *
 * A cold model answers `503` (and starts scaling up, which takes minutes), so
 * it is skipped at once rather than waited on. Anything else that stops a
 * model from giving a full answer, a state too long for it, a timeout, a
 * failed request, malformed JSON or a missing question, skips it too. When
 * every hosted model is skipped the fallback model is asked, if configured.
 * Only when that fails as well does `ask` throw, listing why, so the caller
 * can fall back.
 *
 * Calls are made one at a time, as the endpoint asks, however many batches
 * the caller sends concurrently.
 */
export class CascadeAsker implements JevAsker {
  private readonly models: readonly string[];
  private queue: Promise<unknown> = Promise.resolve();

  constructor(private readonly options: CascadeOptions) {
    this.models = options.apiKey && options.baseUrl ? (options.models ?? []) : [];
    if (this.models.length === 0 && !options.fallback) {
      if (!options.apiKey) throw new Error('decision endpoint API key is not configured');
      if (!options.baseUrl) throw new Error('decision endpoint base URL is not configured');
      throw new Error('no decision models are configured');
    }
  }

  ask(state: JevState, questions: JevQuestions): Promise<JevResponse> {
    const turn = this.queue.then(() => this.cascade(state, questions));
    this.queue = turn.catch(() => undefined);
    return turn;
  }

  private async call(url: string, init: CascadeInit) {
    const ms = this.options.timeoutMs ?? PER_MODEL_TIMEOUT_MS;
    const sleep = this.options.sleep;
    if (!sleep) return this.options.transport(url, init);
    return Promise.race([
      this.options.transport(url, init),
      sleep(ms).then(() => {
        throw new Error(`no answer in ${ms / 1000} s`);
      }),
    ]);
  }

  private async cascade(state: JevState, questions: JevQuestions): Promise<JevResponse> {
    const skipped: string[] = [];
    const names = Object.keys(questions);
    const chars = stateText(state).length;
    for (const model of this.models) {
      const limits = { ...DEFAULT_MODEL_LIMITS, ...this.options.limits?.[model] };
      if (chars > limits.maxStateChars) {
        skipped.push(`${model}: state too long (${chars} > ${limits.maxStateChars} chars)`);
        continue;
      }
      try {
        const answers: Record<string, JevAnswer> = {};
        let reported: string | undefined;
        let problem: string | undefined;
        for (const part of chunk(names, limits.maxQuestions)) {
          const asked = Object.fromEntries(part.map((name) => [name, questions[name]!]));
          const { url, init } = buildCascadeRequest(
            {
              apiKey: this.options.apiKey!,
              baseUrl: this.options.baseUrl!,
              model,
              ...(this.options.auth ? { auth: this.options.auth } : {}),
            },
            state,
            asked,
          );
          const response = await this.call(url, init);
          if (response.status === 503) {
            problem = 'cold';
            break;
          }
          const parsed = parseJevResponse(response.status, response.ok, response.text);
          problem = unusable(parsed, asked);
          if (problem) break;
          Object.assign(answers, parsed.answers);
          if (typeof parsed.model === 'string') reported = parsed.model;
        }
        if (problem) {
          skipped.push(`${model}: ${problem}`);
          continue;
        }
        return { model: reported ?? model, answers, skipped };
      } catch (error) {
        skipped.push(`${model}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    if (this.models.length === 0) skipped.push('hosted chain: no key or base URL');

    const fallback = this.options.fallback;
    const fallbackLimit = fallback?.maxStateChars ?? DEFAULT_FALLBACK_MAX_STATE_CHARS;
    if (fallback && chars > fallbackLimit) {
      skipped.push(`fallback:${fallback.model}: state too long (${chars} > ${fallbackLimit} chars)`);
    } else if (fallback) {
      try {
        return { ...(await this.askFallback(fallback, state, questions)), skipped };
      } catch (error) {
        skipped.push(`fallback:${fallback.model}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    throw new Error(`no decision model answered (${skipped.join('; ')})`);
  }

  /**
   * The fallback model, one chat call per question, all at once (vLLM serves
   * them concurrently). Its confidence is the logprob mass on the answer
   * marker, which is not calibrated the way the hosted models' probabilities
   * are. A reply naming no option, or carrying no logprobs, is no answer: the
   * number decides a deletion, so a guess with no signal is refused and the
   * whole request fails.
   */
  private async askFallback(
    fallback: FallbackEndpoint,
    state: JevState,
    questions: JevQuestions,
  ): Promise<JevResponse> {
    const options = ['yes', 'no'];
    const input = stateText(state);
    const entries = await Promise.all(
      Object.entries(questions).map(async ([name, question]) => {
        if (question.type !== 'noul') throw new Error(`cannot ask ${question.type} questions`);
        const response = await this.call(`${trimBase(fallback.baseUrl)}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            model: fallback.model,
            messages: [
              {
                role: 'user',
                content: `${question.instructions}\n\nInput:\n${input}\n\nAnswer with exactly one of: ${options.join(', ')}. Answer with the single word only, nothing else.`,
              },
            ],
            max_tokens: 16,
            temperature: 0,
            logprobs: true,
            top_logprobs: 20,
            chat_template_kwargs: { enable_thinking: false },
          }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        let choice: ChatChoice | undefined;
        try {
          choice = (JSON.parse(response.text) as { choices?: ChatChoice[] }).choices?.[0];
        } catch {
          throw new Error('malformed JSON');
        }
        const text = stripThinking(choice?.message?.content ?? '').toLowerCase();
        const value = options.find((option) => text.includes(option));
        if (!choice || !value) throw new Error(`no answer for ${name}`);
        const confidence = markerConfidence(choice, options, value);
        if (confidence <= 0) throw new Error(`no confidence for ${name}`);
        return [name, { noul: value === 'yes' ? confidence : 1 - confidence }] as const;
      }),
    );
    return { model: `fallback:${fallback.model}`, answers: Object.fromEntries(entries) };
  }
}

/**
 * Starts cold hosted models scaling up without asking them anything real
 * A status check does not start a model; only a request that
 * finds it cold does, so a model reporting `ready` is left alone and the rest
 * are sent a throwaway question. Failures are ignored: this is a speculative
 * kick, never a dependency. Resolves the models it sent a wake-up.
 */
export async function wakeModels(
  params: { apiKey: string; baseUrl: string; models: readonly string[]; auth?: CascadeAuth },
  transport: CascadeTransport,
): Promise<string[]> {
  const base = trimBase(params.baseUrl);
  const headers = authHeaders(params.apiKey, params.auth);
  const woken = await Promise.all(
    params.models.map(async (model) => {
      try {
        const status = await transport(`${base}/${encodeURIComponent(model)}/status`, {
          method: 'GET',
          headers,
        });
        if (status.ok && (JSON.parse(status.text) as { ready?: unknown }).ready === true) {
          return undefined;
        }
      } catch {
        // Readiness unknown: send the wake-up anyway.
      }
      try {
        const { url, init } = buildCascadeRequest({ ...params, model }, 'wake', {
          wake: { type: 'noul', instructions: 'Is this a wake-up?' },
        });
        await transport(url, init);
      } catch {
        return undefined;
      }
      return model;
    }),
  );
  return woken.filter((model): model is string => model !== undefined);
}
