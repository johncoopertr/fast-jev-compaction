import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  resolveHookConfig,
  resolveRuntimeConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({ compactAtPercent: 60, minReductionRatio: 0.25, model: 'jev-latest' });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

describe('decision endpoint', () => {
  const endpoint = { baseUrl: 'https://decide.test/v1', models: 'cold, warm' };

  it('reads the endpoint options and ignores an unknown auth style', () => {
    expect(resolveHookConfig({ ...endpoint, authHeader: 'bearer' })).toMatchObject({
      baseUrl: 'https://decide.test/v1',
      models: ['cold', 'warm'],
      authHeader: 'bearer',
    });
    expect(resolveHookConfig({ ...endpoint, authHeader: 'basic' }).authHeader).toBeUndefined();
    expect(resolveHookConfig({ baseUrl: '', models: ' , ' })).toEqual(resolveHookConfig({}));
  });

  function host(env: Record<string, string>, settingsEnv: Record<string, string> = {}) {
    return {
      env: { get: async (name: string) => env[name] },
      settings: { read: async () => ({ env: settingsEnv }) },
    };
  }

  it('takes the endpoint and its key from the environment, then from settings', async () => {
    const configured = resolveHookConfig({});
    const fromEnv = await resolveRuntimeConfig(
      host({ DECISION_BASE_URL: 'https://decide.test', DECISION_MODELS: 'a,b', DECISION_API_KEY: 'dk' }),
      configured,
    );
    expect(fromEnv).toMatchObject({ baseUrl: 'https://decide.test', models: ['a', 'b'], apiKey: 'dk' });
    const fromSettings = await resolveRuntimeConfig(
      host({}, { DECISION_BASE_URL: 'https://decide.test', DECISION_MODELS: 'a', DECISION_API_KEY: 'sk' }),
      configured,
    );
    expect(fromSettings).toMatchObject({ models: ['a'], apiKey: 'sk' });
  });

  it('stays on Jev and its key unless both a URL and a model are set', async () => {
    const jev = await resolveRuntimeConfig(
      host({ DECISION_BASE_URL: 'https://decide.test', DECISION_API_KEY: 'dk', TYPESAFE_API_KEY: 'tk' }),
      resolveHookConfig({}),
    );
    expect(jev).toMatchObject({ apiKey: 'tk' });
    expect(jev.baseUrl).toBeUndefined();
    expect(jev.models).toBeUndefined();
  });

  it('asks the models in order over the engine fetch and skips the cold one', async () => {
    const urls: string[] = [];
    const headers: Record<string, string>[] = [];
    const config = {
      ...resolveHookConfig({ ...endpoint, preserveRecentMessages: 1 }),
      apiKey: 'dk',
    };
    const warm = jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1));
    const { result: output } = await compactSession(transcript(), config, async (url, init) => {
      urls.push(url);
      headers.push(init?.headers ?? {});
      if (url.endsWith('/cold')) return { status: 503, ok: false, text: 'scaling' };
      return warm(url, init);
    });
    expect(urls).toEqual(['https://decide.test/v1/cold', 'https://decide.test/v1/warm']);
    expect(headers[1]).toMatchObject({ 'x-api-key': 'dk' });
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
  });

  it('throws when no model answers so the hook falls back to the built-in summary', async () => {
    const config = { ...resolveHookConfig({ ...endpoint, preserveRecentMessages: 1 }), apiKey: 'dk' };
    await expect(
      compactSession(transcript(), config, async () => ({ status: 503, ok: false, text: '' })),
    ).rejects.toThrow(/no decision model answered \(cold: cold; warm: cold\)/);
  });
});
