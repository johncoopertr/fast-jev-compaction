import { describe, expect, it } from 'vitest';
import {
  buildCascadeRequest,
  CascadeAsker,
  fetchTransport,
  fitState,
  markerConfidence,
  parseModelList,
  stripThinking,
  wakeModels,
  type CascadeTransport,
  type JevQuestions,
  type Message,
} from '../src/index.js';

const questions: JevQuestions = {
  call_t1: { type: 'noul', instructions: 'keep the call' },
  result_t1: { type: 'noul', instructions: 'keep the result' },
};

type Seen = { url: string; headers: Record<string, string>; body: unknown };

function answering(noul: number): string {
  return JSON.stringify({
    answers: { call_t1: { noul }, result_t1: { noul } },
  });
}

function scripted(
  replies: Record<string, { status: number; text: string } | Error>,
  seen: Seen[] = [],
): CascadeTransport {
  return async (url, init) => {
    seen.push({ url, headers: init.headers, body: JSON.parse(init.body) });
    const reply = replies[url.split('/').pop() ?? ''];
    if (reply === undefined) throw new Error(`unexpected ${url}`);
    if (reply instanceof Error) throw reply;
    return { ...reply, ok: reply.status >= 200 && reply.status < 300 };
  };
}

function asker(transport: CascadeTransport, models = ['a', 'b', 'c']) {
  return new CascadeAsker({ apiKey: 'k', baseUrl: 'https://decide.test/v1/', models, transport });
}

describe('model list', () => {
  it('splits on commas, trims, and drops blanks and repeats', () => {
    expect(parseModelList(' a, b ,,a,c ')).toEqual(['a', 'b', 'c']);
    expect(parseModelList('')).toEqual([]);
  });
});

describe('cascade request', () => {
  it('puts the model in the path and the key in x-api-key, with no model in the body', () => {
    const { url, init } = buildCascadeRequest(
      { apiKey: 'k', baseUrl: 'https://decide.test/v1//', model: 'a/b' },
      'state',
      questions,
    );
    expect(url).toBe('https://decide.test/v1/a%2Fb');
    expect(init.headers).toEqual({ 'content-type': 'application/json', 'x-api-key': 'k' });
    expect(JSON.parse(init.body)).toEqual({ state: 'state', questions });
  });

  it('can send the key as a bearer token instead', () => {
    const { init } = buildCascadeRequest(
      { apiKey: 'k', baseUrl: 'https://decide.test', model: 'a', auth: 'bearer' },
      {},
      questions,
    );
    expect(init.headers).toEqual({ 'content-type': 'application/json', authorization: 'Bearer k' });
  });
});

describe('CascadeAsker', () => {
  it('returns the first model that answers every question and stops there', async () => {
    const seen: Seen[] = [];
    const response = await asker(
      scripted({ a: { status: 200, text: answering(0.8) } }, seen),
    ).ask({ history: [] }, questions);
    expect(seen.map((s) => s.url)).toEqual(['https://decide.test/v1/a']);
    expect(response.model).toBe('a');
    expect(response.answers).toEqual({ call_t1: { noul: 0.8 }, result_t1: { noul: 0.8 } });
  });

  it('skips a cold model (503) at once and asks the next', async () => {
    const seen: Seen[] = [];
    const response = await asker(
      scripted(
        { a: { status: 503, text: 'scaling' }, b: { status: 200, text: answering(0.3) } },
        seen,
      ),
    ).ask({}, questions);
    expect(seen.map((s) => s.url.split('/').pop())).toEqual(['a', 'b']);
    expect(response.model).toBe('b');
  });

  it('keeps the model name the endpoint reports', async () => {
    const text = JSON.stringify({ model: 'a-v2', answers: { call_t1: { noul: 1 }, result_t1: { noul: 1 } } });
    const response = await asker(scripted({ a: { status: 200, text } })).ask({}, questions);
    expect(response.model).toBe('a-v2');
  });

  it('skips failed requests, thrown errors, malformed JSON and partial answers', async () => {
    const partial = JSON.stringify({ answers: { call_t1: { noul: 0.9 } } });
    const notNumber = JSON.stringify({ answers: { call_t1: { noul: 0.9 }, result_t1: { noul: 'high' } } });
    const response = await asker(
      scripted({
        a: { status: 500, text: 'boom' },
        b: new Error('socket hang up'),
        c: { status: 200, text: '<html>' },
        d: { status: 200, text: partial },
        e: { status: 200, text: notNumber },
        f: { status: 200, text: answering(0.6) },
      }),
      ['a', 'b', 'c', 'd', 'e', 'f'],
    ).ask({}, questions);
    expect(response.model).toBe('f');
  });

  it('throws, naming every reason, when no model answers', async () => {
    const failure = asker(
      scripted({
        a: { status: 503, text: '' },
        b: { status: 200, text: JSON.stringify({ answers: { call_t1: { noul: 1 } } }) },
        c: { status: 401, text: 'forbidden' },
      }),
    ).ask({}, questions);
    await expect(failure).rejects.toThrow(/^no decision model answered \(a: cold; b: no answer for result_t1; c: Jev request failed \(401\)/);
  });

  it('skips a model whose state limit the state exceeds, without asking it', async () => {
    const seen: Seen[] = [];
    const response = await new CascadeAsker({
      apiKey: 'k',
      baseUrl: 'https://decide.test',
      models: ['short', 'long'],
      limits: { short: { maxStateChars: 15_000 } },
      transport: scripted({ long: { status: 200, text: answering(0.7) } }, seen),
    }).ask({ history: 'x'.repeat(20_000) }, questions);
    expect(seen.map((s) => s.url.split('/').pop())).toEqual(['long']);
    expect(response.skipped).toEqual(['short: state too long (20014 > 15000 chars)']);
  });

  it("splits the questions to each model's cap and merges the answers", async () => {
    const many: JevQuestions = Object.fromEntries(
      Array.from({ length: 20 }, (_, i) => [`q${i}`, { type: 'noul', instructions: `q${i}` }]),
    );
    const sizes: Record<string, number[]> = {};
    const transport: CascadeTransport = async (url, init) => {
      const model = url.split('/').pop()!;
      const asked = Object.keys((JSON.parse(init.body!) as { questions: object }).questions);
      (sizes[model] ??= []).push(asked.length);
      if (model === 'other') return { status: 413, ok: false, text: 'too many' };
      const answers = Object.fromEntries(asked.map((name) => [name, { noul: 0.5 }]));
      return { status: 200, ok: true, text: JSON.stringify({ answers }) };
    };
    const response = await new CascadeAsker({
      apiKey: 'k',
      baseUrl: 'https://decide.test',
      models: ['other', 'wide'],
      limits: { wide: { maxQuestions: 16 } },
      transport,
    }).ask({}, many);
    expect(sizes).toEqual({ other: [8], wide: [16, 4] });
    expect(Object.keys(response.answers)).toHaveLength(20);
  });

  it('moves on when a model does not answer in time', async () => {
    const transport: CascadeTransport = (url) =>
      url.endsWith('/slow')
        ? new Promise(() => undefined)
        : Promise.resolve({ status: 200, ok: true, text: answering(0.4) });
    const response = await new CascadeAsker({
      apiKey: 'k',
      baseUrl: 'https://decide.test',
      models: ['slow', 'fast'],
      transport,
      timeoutMs: 5,
      sleep: () => Promise.resolve(),
    }).ask({}, questions);
    expect(response.model).toBe('fast');
    expect(response.skipped).toEqual(['slow: no answer in 0.005 s']);
  });

  it('sends one request at a time however many batches are in flight', async () => {
    let inFlight = 0;
    let most = 0;
    const transport: CascadeTransport = async () => {
      most = Math.max(most, ++inFlight);
      await new Promise((done) => setTimeout(done, 5));
      inFlight--;
      return { status: 200, ok: true, text: answering(0.5) };
    };
    const cascade = asker(transport, ['a']);
    await Promise.all([cascade.ask({}, questions), cascade.ask({}, questions), cascade.ask({}, questions)]);
    expect(most).toBe(1);
  });

  it('refuses to start without a key, a URL or a model', () => {
    const transport = scripted({});
    const base = { apiKey: 'k', baseUrl: 'https://decide.test', models: ['a'], transport };
    expect(() => new CascadeAsker({ ...base, apiKey: '' })).toThrow(/API key/);
    expect(() => new CascadeAsker({ ...base, baseUrl: '' })).toThrow(/base URL/);
    expect(() => new CascadeAsker({ ...base, models: [] })).toThrow(/no decision models/);
  });
});

describe('fallback model', () => {
  function chat(content: string, top: [string, number][] | undefined) {
    return JSON.stringify({
      choices: [
        {
          message: { content },
          ...(top
            ? { logprobs: { content: [{ token: top[0]![0], top_logprobs: top.map(([token, p]) => ({ token, logprob: Math.log(p) })) }] } }
            : {}),
        },
      ],
    });
  }

  const fallback = { baseUrl: 'http://fallback.test/', model: 'Qwen/Qwen3.5-4B' };

  it('answers when every hosted model is skipped, one chat call per question', async () => {
    const seen: Seen[] = [];
    const transport: CascadeTransport = async (url, init) => {
      seen.push({ url, headers: init.headers, body: JSON.parse(init.body!) });
      if (!url.endsWith('/chat/completions')) return { status: 503, ok: false, text: '' };
      return { status: 200, ok: true, text: chat('No', [['No', 0.6], ['Yes', 0.2], ['.', 0.2]]) };
    };
    const response = await new CascadeAsker({
      apiKey: 'k',
      baseUrl: 'https://decide.test',
      models: ['a'],
      fallback,
      transport,
    }).ask({ s: 1 }, questions);
    expect(response.model).toBe('fallback:Qwen/Qwen3.5-4B');
    expect(response.skipped).toEqual(['a: cold']);
    expect((response.answers.call_t1 as { noul: number }).noul).toBeCloseTo(0.25);
    const asked = seen.slice(1);
    expect(asked.map((s) => s.url)).toEqual([
      'http://fallback.test/v1/chat/completions',
      'http://fallback.test/v1/chat/completions',
    ]);
    expect(asked[0]!.body).toMatchObject({
      model: 'Qwen/Qwen3.5-4B',
      max_tokens: 16,
      temperature: 0,
      logprobs: true,
      top_logprobs: 20,
      chat_template_kwargs: { enable_thinking: false },
      messages: [
        {
          role: 'user',
          content:
            'keep the call\n\nInput:\n{"s":1}\n\nAnswer with exactly one of: yes, no. Answer with the single word only, nothing else.',
        },
      ],
    });
  });

  it('is asked alone when there is no decision key', async () => {
    const urls: string[] = [];
    const response = await new CascadeAsker({
      fallback,
      transport: async (url) => {
        urls.push(url);
        return { status: 200, ok: true, text: chat('yes', [['yes', 1]]) };
      },
    }).ask({}, questions);
    expect(response.answers.result_t1).toEqual({ noul: 1 });
    expect(urls.every((url) => url.startsWith('http://fallback.test/'))).toBe(true);
  });

  it('refuses a reply with no logprobs or no option, so nothing is deleted on a guess', async () => {
    for (const text of [chat('yes', undefined), chat('maybe', [['maybe', 1]])]) {
      const failure = new CascadeAsker({
        fallback,
        transport: async () => ({ status: 200, ok: true, text }),
      }).ask({}, questions);
      await expect(failure).rejects.toThrow(/no decision model answered \(hosted chain: no key or base URL; fallback:/);
    }
  });

  it('is skipped without a request when the state is longer than its context', async () => {
    let calls = 0;
    const failure = new CascadeAsker({
      fallback: { ...fallback, maxStateChars: 10 },
      transport: async () => {
        calls++;
        return { status: 200, ok: true, text: chat('yes', [['yes', 1]]) };
      },
    }).ask({ history: 'x'.repeat(20) }, questions);
    await expect(failure).rejects.toThrow(/fallback:Qwen\/Qwen3\.5-4B: state too long \(34 > 10 chars\)/);
    expect(calls).toBe(0);
  });

  it('reads the answer after any thinking block', () => {
    expect(stripThinking('<think>not yes, surely</think>\n\nno').trim()).toBe('no');
    expect(stripThinking('<think>yes')).toBe('');
  });

  it('weighs only the option markers', () => {
    const choice = {
      logprobs: {
        content: [
          { token: 'The', top_logprobs: [] },
          { token: ' Yes', top_logprobs: [{ token: 'Yes', logprob: Math.log(0.3) }, { token: 'no', logprob: Math.log(0.1) }, { token: 'x', logprob: Math.log(0.6) }] },
        ],
      },
    };
    expect(markerConfidence(choice, ['yes', 'no'], 'yes')).toBeCloseTo(0.75);
    expect(markerConfidence({}, ['yes', 'no'], 'yes')).toBe(0);
  });
});

describe('wake-up', () => {
  it('leaves ready models alone and sends the rest a throwaway question', async () => {
    const seen: { method: string; url: string }[] = [];
    const woken = await wakeModels(
      { apiKey: 'k', baseUrl: 'https://decide.test/v1/', models: ['warm', 'cold', 'down'] },
      async (url, init) => {
        seen.push({ method: init.method, url });
        if (url.endsWith('/warm/status')) return { status: 200, ok: true, text: '{"ready":true}' };
        if (url.endsWith('/cold/status')) return { status: 200, ok: true, text: '{"ready":false}' };
        if (url.endsWith('/down/status')) throw new Error('unreachable');
        return { status: 503, ok: false, text: '' };
      },
    );
    expect(woken).toEqual(['cold', 'down']);
    expect(seen.filter((s) => s.method === 'POST').map((s) => s.url)).toEqual([
      'https://decide.test/v1/cold',
      'https://decide.test/v1/down',
    ]);
  });
});

describe('state character budget', () => {
  it('shrinks the state until it fits maxStateChars too', () => {
    const messages: Message[] = Array.from({ length: 12 }, (_, i) => ({
      role: i % 2 ? 'assistant' : 'user',
      text: `message ${i} `.repeat(200),
      toolUses: [],
    }));
    const options = { maxStateTokens: 1_000_000, preserveRecentMessages: 2, goal: 'g' };
    const full = fitState(messages, [], options);
    const fitted = fitState(messages, [], { ...options, maxStateChars: 8_000 });
    expect(JSON.stringify(full.state).length).toBeGreaterThan(8_000);
    expect(JSON.stringify(fitted.state).length).toBeLessThanOrEqual(8_000);
    expect(fitted.stage).not.toBe('full');
    expect(() => fitState(messages, [], { ...options, maxStateChars: 100 })).toThrow(/100 chars/);
  });
});

describe('fetch transport', () => {
  it('reads the body so the cascade sees status, ok and text', async () => {
    const transport = fetchTransport((async () =>
      new Response('hello', { status: 503 })) as typeof fetch);
    expect(
      await transport('https://decide.test/a', { method: 'POST', headers: {}, body: '{}' }),
    ).toEqual({ status: 503, ok: false, text: 'hello' });
  });
});
