import { parseJevResponse } from './request.js';
import type { JevAsker, JevQuestions, JevResponse, JevState } from './types.js';

export type CascadeAuth = 'x-api-key' | 'bearer';

export type CascadeInit = {
  method: 'POST';
  headers: Record<string, string>;
  body: string;
};

/** A fetch that has already read the body: `$.http.fetch` in the hook, `fetchTransport` elsewhere. */
export type CascadeTransport = (
  url: string,
  init: CascadeInit,
) => Promise<{ status: number; ok: boolean; text: string }>;

export interface CascadeOptions {
  apiKey: string;
  /** The decision endpoint; each model is asked at `{baseUrl}/{model}`. */
  baseUrl: string;
  /** Models to try, best first. */
  models: readonly string[];
  /** How the key is sent. Default `x-api-key`. */
  auth?: CascadeAuth;
  transport: CascadeTransport;
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

/** The HTTP request for one model of the cascade. The model rides in the path, not the body. */
export function buildCascadeRequest(
  params: { apiKey: string; baseUrl: string; model: string; auth?: CascadeAuth },
  state: JevState,
  questions: JevQuestions,
): { url: string; init: CascadeInit } {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (params.auth === 'bearer') headers['authorization'] = `Bearer ${params.apiKey}`;
  else headers['x-api-key'] = params.apiKey;
  return {
    url: `${params.baseUrl.replace(/\/+$/, '')}/${encodeURIComponent(params.model)}`,
    init: { method: 'POST', headers, body: JSON.stringify({ state, questions }) },
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

/**
 * Asks a chain of decision models and returns the first complete answer.
 *
 * A model that is cold answers `503` (and starts scaling up, which takes
 * minutes), so it is skipped at once rather than waited on. Anything else that
 * stops a model from giving a full answer, a failed request, malformed JSON or
 * a missing question, skips it too. Only when every model has been skipped does
 * `ask` throw, listing why, so the caller can fall back.
 */
export class CascadeAsker implements JevAsker {
  private readonly models: readonly string[];

  constructor(private readonly options: CascadeOptions) {
    this.models = options.models;
    if (!options.apiKey) throw new Error('decision endpoint API key is not configured');
    if (!options.baseUrl) throw new Error('decision endpoint base URL is not configured');
    if (this.models.length === 0) throw new Error('no decision models are configured');
  }

  async ask(state: JevState, questions: JevQuestions): Promise<JevResponse> {
    const skipped: string[] = [];
    for (const model of this.models) {
      const { url, init } = buildCascadeRequest(
        { ...this.options, model },
        state,
        questions,
      );
      try {
        const response = await this.options.transport(url, init);
        if (response.status === 503) {
          skipped.push(`${model}: cold`);
          continue;
        }
        const parsed = parseJevResponse(response.status, response.ok, response.text);
        const problem = unusable(parsed, questions);
        if (problem) {
          skipped.push(`${model}: ${problem}`);
          continue;
        }
        return { ...parsed, model: typeof parsed.model === 'string' ? parsed.model : model };
      } catch (error) {
        skipped.push(`${model}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    throw new Error(`no decision model answered (${skipped.join('; ')})`);
  }
}
