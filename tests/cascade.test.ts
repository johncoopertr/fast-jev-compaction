import { describe, expect, it } from 'vitest';
import {
  buildCascadeRequest,
  CascadeAsker,
  fetchTransport,
  parseModelList,
  type CascadeTransport,
  type JevQuestions,
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

  it('refuses to start without a key, a URL or a model', () => {
    const transport = scripted({});
    const base = { apiKey: 'k', baseUrl: 'https://decide.test', models: ['a'], transport };
    expect(() => new CascadeAsker({ ...base, apiKey: '' })).toThrow(/API key/);
    expect(() => new CascadeAsker({ ...base, baseUrl: '' })).toThrow(/base URL/);
    expect(() => new CascadeAsker({ ...base, models: [] })).toThrow(/no decision models/);
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
