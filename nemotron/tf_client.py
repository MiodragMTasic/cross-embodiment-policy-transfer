"""Thin Nebius Token Factory client with a hard spend cap, a spend ledger and a dry-run mode.

Facts (checked 24 Oct 2026, snapshots with sha256 in docs_snapshot/):
  base URL https://api.tokenfactory.nebius.com/v1/ (OpenAI-compatible), credential env var NEBIUS_API_KEY;
  structured output = response_format {"type": "json_schema", "json_schema": {"name", "schema", "strict"}}, looser
  {"type": "json_object"}; model ids and prices from the public models_info catalog. None of the Nemotron catalog
  entries carries a JSON-mode flag, so every response is validated locally and schema enforcement is unverified until
  the smoke test.
Lightning thinking control: chat_template_kwargs.enable_thinking=false (Nebius Workbench verification: without it one
  call spent 506 of 512 output tokens on reasoning and returned truncated JSON). Sent by default.

Key: NEBIUS_API_KEY, else ~/.config/nebius/token-factory.key, read in the constructor only, never in dry-run, never
  written, printed or logged; every error string passes through redact().
Spend cap: SPEND_CAP_USD over the whole ledger runs/spend.jsonl. A call is refused when spent + its worst case
  (estimated prompt tokens + max_tokens, uncached prices) would pass the cap. No env or CLI override.

  python tf_client.py --dry-run     print the smoke-test request and its estimated cost; no call, no key read
  python tf_client.py --smoke       exactly one real Lightning call with json_schema, no retry (only after
                                    "smoke test GO"); saved to runs/smoke/
  python tf_client.py --spent       ledger total
"""
import argparse
import fcntl
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, append_jsonl, write_json  # noqa: E402

try:
    import jsonschema
except ImportError:  # hand validator below covers the subset our schemas use
    jsonschema = None

BASE_URL = 'https://api.tokenfactory.nebius.com/v1/'
KEY_FILE = '~/.config/nebius/token-factory.key'
SPEND_CAP_USD = 2.00
LEDGER_PATH = RUNS / 'spend.jsonl'
LIGHTNING = 'nvidia/Nemotron-3_5-Lightning'
PRICES = {  # USD per 1M tokens (input, output), uncached; docs_snapshot/models_info.json
    'nvidia/Nemotron-3_5-Lightning': (0.06, 0.24),
    'nvidia/Nemotron-3.5-Super-VL': (0.30, 0.90),
    'nvidia/nemotron-3-super-120b-a12b': (0.30, 0.90),
    'nvidia/Nemotron-3-Ultra-550b-a55b': (1.00, 3.00),
}
TIMEOUT_S = 120


class SpendCapExceeded(RuntimeError):
    pass


class TransportError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f'HTTP {status}: {message}')
        self.status = status


def estimate_tokens(text):
    """Upper-leaning estimate (3 chars per token); the ledger uses the token counts the response reports."""
    return math.ceil(len(text) / 3)


def usd(model, prompt_tokens, completion_tokens):
    p_in, p_out = PRICES[model]
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


def ledger_rows(path=None):
    path = Path(path or LEDGER_PATH)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def spent(path=None):
    return sum(r['usd'] for r in ledger_rows(path))


def estimate(payload):
    """Prompt-token estimate covers the messages plus the schema (in case the server injects it)."""
    text = ''.join(m['content'] for m in payload['messages']) + json.dumps(payload.get('response_format', ''))
    prompt = estimate_tokens(text) + 8 * len(payload['messages'])
    likely_out = min(payload['max_tokens'], 400)
    return {'prompt_tokens_est': prompt, 'max_completion_tokens': payload['max_tokens'],
            'worst_usd': usd(payload['model'], prompt, payload['max_tokens']),
            'likely_usd': usd(payload['model'], prompt, likely_out)}


def validate(schema, data):
    """Errors as strings; empty means valid. jsonschema when installed, else a hand validator for our subset."""
    if jsonschema is not None:
        return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
                for e in jsonschema.Draft202012Validator(schema).iter_errors(data)]
    return _hand_validate(schema, data, '<root>')


def _hand_validate(s, x, at):
    t = s.get('type')
    kinds = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'integer': int, 'number': (int, float)}
    if t and (not isinstance(x, kinds[t]) or (t in ('integer', 'number') and isinstance(x, bool))):
        return [f'{at}: expected {t}']
    errs = []
    if 'enum' in s and x not in s['enum']:
        errs.append(f'{at}: {x!r} not in enum')
    if t == 'object':
        errs += [f'{at}: missing {k}' for k in s.get('required', []) if k not in x]
        if s.get('additionalProperties') is False:
            errs += [f'{at}: unexpected {k}' for k in x if k not in s.get('properties', {})]
        for k, sub in s.get('properties', {}).items():
            if k in x:
                errs += _hand_validate(sub, x[k], f'{at}/{k}')
    if t == 'array':
        if len(x) > s.get('maxItems', len(x)) or len(x) < s.get('minItems', 0):
            errs.append(f'{at}: length {len(x)} out of range')
        for i, v in enumerate(x):
            errs += _hand_validate(s.get('items', {}), v, f'{at}/{i}')
    if t in ('integer', 'number'):
        if 'minimum' in s and x < s['minimum'] or 'maximum' in s and x > s['maximum']:
            errs.append(f'{at}: {x} out of range')
    if t == 'string' and len(x) > s.get('maxLength', len(x)):
        errs.append(f'{at}: string longer than {s["maxLength"]}')
    return errs


def build_request(model, messages, schema=None, name='output', mode='json_schema', max_tokens=800, temperature=0.0,
                  thinking=False):
    """The exact JSON body sent to POST {BASE_URL}chat/completions."""
    if model not in PRICES:
        raise ValueError(f'unknown model id {model!r}; ids are case-sensitive: {sorted(PRICES)}')
    body = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature}
    if mode == 'json_schema':
        body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': name, 'schema': schema, 'strict': True}}
    elif mode == 'json_object':
        body['response_format'] = {'type': 'json_object'}
    elif mode != 'prompted':
        raise ValueError(mode)
    if not thinking:
        body['chat_template_kwargs'] = {'enable_thinking': False}
    return body


class TFClient:
    def __init__(self, dry_run=False, transport=None):
        self.dry_run = dry_run
        self._key = None
        if not dry_run:
            self._key = os.environ.get('NEBIUS_API_KEY') or Path(KEY_FILE).expanduser().read_text().strip()
            if not self._key:
                raise RuntimeError('no Token Factory credential: set NEBIUS_API_KEY or save the key file')
        self._transport = transport or (None if dry_run else self._default_transport())

    def __repr__(self):
        return f'TFClient(dry_run={self.dry_run}, key=<redacted>)'

    def redact(self, text):
        text = str(text)
        if self._key:
            text = text.replace(self._key, '[REDACTED]')
        return text

    def _default_transport(self):
        import importlib.util
        return self._openai_transport if importlib.util.find_spec('openai') else self._urllib_transport

    def _openai_transport(self, payload):
        import openai
        known = ('model', 'messages', 'max_tokens', 'temperature', 'response_format')
        client = openai.OpenAI(base_url=BASE_URL, api_key=self._key, max_retries=0, timeout=TIMEOUT_S)
        try:
            resp = client.chat.completions.create(**{k: payload[k] for k in known if k in payload},
                                                  extra_body={k: v for k, v in payload.items() if k not in known} or None)
        except openai.APIStatusError as e:
            raise TransportError(e.status_code, self.redact(e.message)) from None
        except Exception as e:  # connection errors, timeouts, anything the SDK raises
            raise TransportError(None, self.redact(f'{type(e).__name__}: {e}')) from None
        return resp.model_dump()

    def _urllib_transport(self, payload):
        req = urllib.request.Request(BASE_URL + 'chat/completions', data=json.dumps(payload).encode(), method='POST',
                                     headers={'Authorization': f'Bearer {self._key}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise TransportError(e.code, self.redact(e.read().decode(errors='replace')[:2000])) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise TransportError(None, self.redact(f'{type(e).__name__}: {e}')) from None

    def describe(self, payload):
        """Dry-run text: endpoint, headers with the key elided, exact body, cost estimate."""
        est = estimate(payload)
        return '\n'.join([
            f'POST {BASE_URL}chat/completions',
            'Authorization: Bearer <NEBIUS_API_KEY, loaded at runtime, never printed>',
            'Content-Type: application/json', '', json.dumps(payload, indent=1), '',
            f"estimate: ~{est['prompt_tokens_est']} prompt tokens (chars/3), <= {est['max_completion_tokens']} completion; "
            f"likely ${est['likely_usd']:.6f}, worst ${est['worst_usd']:.6f}; ledger spent ${spent():.6f} of cap ${SPEND_CAP_USD:.2f}"])

    def complete(self, payload, purpose):
        """One network call under the cap. Returns the raw response dict; raises SpendCapExceeded or TransportError."""
        if self.dry_run:
            raise RuntimeError('complete() called in dry-run mode')
        est = estimate(payload)
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH.with_suffix('.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # check, call and record are one critical section across processes
            already = spent()
            if already + est['worst_usd'] > SPEND_CAP_USD:
                raise SpendCapExceeded(f"refused: spent ${already:.6f} + worst case ${est['worst_usd']:.6f} "
                                       f'> cap ${SPEND_CAP_USD:.2f}')
            row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'model': payload['model'], 'purpose': purpose}
            t0 = time.time()
            try:
                resp = self._transport(payload)
            except Exception as e:
                if not isinstance(e, TransportError):
                    e = TransportError(None, self.redact(f'{type(e).__name__}: {e}'))
                timeout = e.status is None  # a dropped connection may still be billed: charge the worst case
                append_jsonl(LEDGER_PATH, {**row, 'status': 'transport_error', 'http_status': e.status,
                                           'prompt_tokens': est['prompt_tokens_est'] if timeout else 0,
                                           'completion_tokens': payload['max_tokens'] if timeout else 0,
                                           'usd': est['worst_usd'] if timeout else 0.0,
                                           'usage_source': 'worst_case_estimate' if timeout else 'none_http_error',
                                           'error': self.redact(e)[:300]})
                raise e from None
            u = resp.get('usage') or {}
            have = isinstance(u.get('prompt_tokens'), int) and isinstance(u.get('completion_tokens'), int)
            pt, ct = (u['prompt_tokens'], u['completion_tokens']) if have else (est['prompt_tokens_est'], payload['max_tokens'])
            details = u.get('completion_tokens_details') or {}
            append_jsonl(LEDGER_PATH, {**row, 'status': 'ok', 'served_model': resp.get('model'), 'response_id': resp.get('id'),
                                       'prompt_tokens': pt, 'completion_tokens': ct,
                                       'reasoning_tokens': details.get('reasoning_tokens'), 'usd': usd(payload['model'], pt, ct),
                                       'usage_source': 'response' if have else 'worst_case_estimate',
                                       'latency_s': round(time.time() - t0, 3)})
            return resp

    def chat_json(self, payload, schema, purpose, check=None):
        """Call, parse, validate; one retry on refusal, truncation, malformed or invalid JSON, or an HTTP 400 on
        json_schema (retried as json_object). Never raises for model output; returns a record of every attempt."""
        if self.dry_run:
            return {'ok': None, 'dry_run': True, 'estimate': estimate(payload), 'attempts': []}
        attempts, current = [], payload
        for n in (1, 2):
            a = {'n': n, 'mode': (current.get('response_format') or {}).get('type', 'prompted')}
            try:
                resp = self.complete(current, purpose)
            except TransportError as e:
                a.update(error=f'transport: {self.redact(e)[:500]}', http_status=e.status)
                attempts.append(a)
                if n == 1 and e.status == 400 and a['mode'] == 'json_schema':
                    current = {**current, 'response_format': {'type': 'json_object'}}
                    continue
                if n == 1 and e.status in (None, 429, 500, 502, 503, 504):
                    time.sleep(5)
                    continue
                break
            choice = (resp.get('choices') or [{}])[0]
            msg = choice.get('message') or {}
            a.update(response=resp, finish_reason=choice.get('finish_reason'))
            content = msg.get('content') or ''
            if msg.get('refusal'):
                a['error'] = f"refusal: {msg['refusal']}"
            elif choice.get('finish_reason') == 'length':
                a['error'] = 'truncated: finish_reason=length'
            else:
                try:
                    data = json.loads(content)
                    errs = validate(schema, data) + (check(data) if check else [])
                    a['error'] = '; '.join(errs)[:1000] if errs else None
                    if not errs:
                        attempts.append(a)
                        return {'ok': True, 'data': data, 'attempts': attempts}
                except json.JSONDecodeError as e:
                    a['error'] = f'malformed JSON: {e}'
            attempts.append(a)
            if a['error'].startswith('truncated'):
                current = {**current, 'max_tokens': current['max_tokens'] * 2}
            else:
                current = {**current, 'messages': current['messages'] + [
                    {'role': 'assistant', 'content': content[:4000]},
                    {'role': 'user', 'content': f"That reply failed validation ({a['error'][:300]}). "
                                                'Return only one JSON object that matches the schema.'}]}
        return {'ok': False, 'error': attempts[-1].get('error'), 'attempts': attempts}


SMOKE_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['is_fruit', 'colour_code'],
                'properties': {'is_fruit': {'type': 'boolean'}, 'colour_code': {'type': 'integer', 'minimum': 1, 'maximum': 9}}}


def smoke_request():
    """Tiny prompt that does not describe the schema, so conforming output is evidence the schema was applied."""
    return build_request(LIGHTNING, [{'role': 'user', 'content': 'Is a banana a fruit? Answer in JSON.'}],
                         SMOKE_SCHEMA, name='smoke', max_tokens=128)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--smoke', action='store_true')
    g.add_argument('--spent', action='store_true')
    a = ap.parse_args(argv)
    if a.spent:
        rows = ledger_rows()
        print(f'{len(rows)} ledger rows, ${spent():.6f} of ${SPEND_CAP_USD:.2f}')
        return 0
    payload = smoke_request()
    if a.dry_run:
        print(TFClient(dry_run=True).describe(payload))
        return 0
    c = TFClient()
    try:  # exactly one call, no retry: a retry would blur whether the schema itself was enforced
        resp = c.complete(payload, 'smoke')
    except TransportError as e:
        print(json.dumps({'ok': False, 'error': c.redact(e), 'ledger_spent_usd': round(spent(), 6)}, indent=1))
        return 1
    choice = (resp.get('choices') or [{}])[0]
    msg = choice.get('message') or {}
    try:
        data, errs = json.loads(msg.get('content') or ''), None
        errs = validate(SMOKE_SCHEMA, data)
    except json.JSONDecodeError as e:
        data, errs = None, [f'malformed JSON: {e}']
    out = {'ok': not errs, 'schema_errors': errs, 'content': msg.get('content'), 'parsed': data,
           'finish_reason': choice.get('finish_reason'), 'refusal': msg.get('refusal'),
           'reasoning_present': bool(msg.get('reasoning_content') or msg.get('reasoning')),
           'served_model': resp.get('model'), 'usage': resp.get('usage'), 'message_keys': sorted(msg),
           'response_keys': sorted(resp), 'ledger_spent_usd': round(spent(), 6)}
    write_json(LEDGER_PATH.parent / 'smoke' / time.strftime('%Y%m%dT%H%M%SZ.json', time.gmtime()), {'request': payload, 'response': resp, 'summary': out})
    print(json.dumps(out, indent=1))
    return 0 if out['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
