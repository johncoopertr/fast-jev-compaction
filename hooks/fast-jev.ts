import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import {
  CascadeAsker,
  DEFAULT_FALLBACK_MODEL,
  parseModelList,
  wakeModels,
  type CascadeAuth,
  type FallbackEndpoint,
} from '../src/cascade.js';
import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  wakeAtPercent: 40,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

/**
 * State budgets for the decision models when none are configured: under the
 * 50,000 characters and ~16,000 tokens a typical decision model takes before
 * it truncates without saying so.
 */
export const CASCADE_BUDGETS = {
  maxStateChars: 48_000,
  maxStateTokens: 12_000,
  maxRequestTokens: 14_000,
};

/** A wake-up repeated inside a model's minutes-long start achieves nothing. */
const WAKE_EVERY_MS = 5 * 60_000;

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  /** With `models`, asks that decision endpoint's models in order instead of Jev. */
  baseUrl?: string;
  models?: string[];
  authHeader?: CascadeAuth;
  /** The fallback model, asked when every hosted model is skipped. */
  fallback?: FallbackEndpoint;
  compactAtPercent: number;
  /** Context percentage at which cold decision models are sent a wake-up. */
  wakeAtPercent: number;
  minReductionRatio: number;
  model: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxStateChars',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    wakeAtPercent: optionNumber(options, 'wakeAtPercent', HOOK_DEFAULTS.wakeAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const baseUrl = optionString(options, 'baseUrl');
  if (baseUrl) config.baseUrl = baseUrl;
  const modelList = optionString(options, 'models');
  const models = modelList ? parseModelList(modelList) : [];
  if (models.length > 0) config.models = models;
  const authHeader = optionString(options, 'authHeader');
  if (authHeader === 'x-api-key' || authHeader === 'bearer') config.authHeader = authHeader;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

function isCascade(config: HookConfig): boolean {
  return Boolean(config.fallback) || Boolean(config.baseUrl && config.models && config.models.length > 0);
}

function askerFor(
  config: HookConfig,
  fetchFn: HookFetch,
  sleep?: (ms: number) => Promise<void>,
): JevAsker {
  if (isCascade(config)) {
    return new CascadeAsker({
      ...(config.apiKey ? { apiKey: config.apiKey } : {}),
      ...(config.baseUrl ? { baseUrl: config.baseUrl } : {}),
      ...(config.models ? { models: config.models } : {}),
      ...(config.authHeader ? { auth: config.authHeader } : {}),
      ...(config.fallback ? { fallback: config.fallback } : {}),
      ...(sleep ? { sleep } : {}),
      transport: (url, init) => fetchFn(url, init),
    });
  }
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  return jevAsker(fetchFn, config.apiKey, config.model);
}

/** Runs the library over a session transcript; throws when the key is missing or no model answers. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  sleep?: (ms: number) => Promise<void>,
): Promise<SessionCompaction> {
  const result = await compact(messages, askerFor(config, fetchFn, sleep), config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  const models = Object.entries(stats.models)
    .map(([model, requests]) => (requests > 1 ? `${model} x${requests}` : model))
    .join(', ');
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)${
    models ? ` by ${models}` : ''
  }`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

type Lookup = {
  env: { get: (name: string) => Promise<string | undefined> };
  settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
};

/** `env` in the Claude Code settings, which backs up the process environment. */
async function settingsEnv($: Lookup): Promise<Record<string, string>> {
  const env = (await $.settings.read())['env'];
  if (!env || typeof env !== 'object') return {};
  return Object.fromEntries(
    Object.entries(env).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
  );
}

/**
 * Fills what the plugin options left open, choosing between a decision-model
 * cascade and Jev.
 *
 * The hosted chain is DECISION_BASE_URL with DECISION_MODELS (a comma list,
 * best first) and DECISION_API_KEY, or the `baseUrl`/`models`/`apiKey`
 * options. DECISION_FALLBACK_URL adds an OpenAI-compatible fallback model
 * (DECISION_FALLBACK_MODEL, default `Qwen/Qwen3.5-4B`), which also works
 * alone. Either one switches from Jev, with state budgets a decision model
 * will not truncate. Otherwise the key is TYPESAFE_API_KEY and Jev is asked.
 */
export async function resolveRuntimeConfig($: Lookup, configured: HookConfig): Promise<HookConfig> {
  // $.env.get takes a literal name so the host can list what a module reads.
  const env = {
    DECISION_BASE_URL: await $.env.get('DECISION_BASE_URL'),
    DECISION_MODELS: await $.env.get('DECISION_MODELS'),
    DECISION_API_KEY: await $.env.get('DECISION_API_KEY'),
    DECISION_FALLBACK_URL: await $.env.get('DECISION_FALLBACK_URL'),
    DECISION_FALLBACK_MODEL: await $.env.get('DECISION_FALLBACK_MODEL'),
    TYPESAFE_API_KEY: await $.env.get('TYPESAFE_API_KEY'),
  };
  const backup = await settingsEnv($);
  const get = (name: keyof typeof env): string | undefined =>
    env[name]?.trim() || backup[name]?.trim() || undefined;

  const config: HookConfig = { ...configured };
  const baseUrl = configured.baseUrl ?? get('DECISION_BASE_URL');
  const models = configured.models ?? parseModelList(get('DECISION_MODELS') ?? '');
  const fallbackUrl = get('DECISION_FALLBACK_URL');
  const hosted = Boolean(baseUrl) && models.length > 0;
  if (!hosted && !fallbackUrl) {
    const apiKey = configured.apiKey ?? get('TYPESAFE_API_KEY');
    if (apiKey) config.apiKey = apiKey;
    return config;
  }

  if (hosted && baseUrl) {
    config.baseUrl = baseUrl;
    config.models = models;
    const apiKey = configured.apiKey ?? get('DECISION_API_KEY');
    if (apiKey) config.apiKey = apiKey;
  }
  if (fallbackUrl) {
    config.fallback = {
      baseUrl: fallbackUrl,
      model: get('DECISION_FALLBACK_MODEL') ?? DEFAULT_FALLBACK_MODEL,
    };
  }
  config.maxStateChars ??= CASCADE_BUDGETS.maxStateChars;
  config.maxStateTokens ??= CASCADE_BUDGETS.maxStateTokens;
  config.maxRequestTokens ??= CASCADE_BUDGETS.maxRequestTokens;
  return config;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;
  let lastWake = Number.NEGATIVE_INFINITY;

  on('session.compact', async ($, event, next) => {
    try {
      const config = await resolveRuntimeConfig($, configured);
      const { result, messages } = await compactSession(
        event.messages,
        config,
        async (url, init) => {
          const response = await $.http.fetch(url, init);
          return { status: response.status, ok: response.ok, text: response.text };
        },
        (ms) => $.clock.sleep(ms, { signal: next.signal }),
      );
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        notify(
          $,
          `fallback to built-in summary (below ${percent(config.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        return next(event);
      }
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      notify(
        $,
        `fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`,
      );
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      const used = context.percent ?? 0;
      if (used >= configured.wakeAtPercent) {
        // Models take 7-15 minutes to come up from zero; start them before
        // compaction needs them rather than when it does.
        const now = await $.clock.now();
        const config = await resolveRuntimeConfig($, configured);
        if (now - lastWake >= WAKE_EVERY_MS && config.apiKey && config.baseUrl && config.models) {
          lastWake = now;
          const woken = await wakeModels(
            {
              apiKey: config.apiKey,
              baseUrl: config.baseUrl,
              models: config.models,
              ...(config.authHeader ? { auth: config.authHeader } : {}),
            },
            async (url, init) => {
              const response = await $.http.fetch(url, init);
              return { status: response.status, ok: response.ok, text: response.text };
            },
          );
          if (woken.length > 0) $.ui.log(`sent a wake-up to ${woken.join(', ')}`);
        }
      }
      if (used < configured.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
