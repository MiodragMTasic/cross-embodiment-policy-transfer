"""Offline accounting of one copied run's judge traffic and executed gripper commands.
Counts what happened; it is NOT the outcome score (that stays the frozen v5.1 scorer on the executed command).
Reads only runtime logs: drop/, perception/judge_events.jsonl and gate_trace.jsonl command fields.
"""
import argparse,json
from collections import Counter
from pathlib import Path


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def score(run):
    queries=[];errors=Counter();phases=Counter()
    for folder in sorted((run/'drop').glob('query_*')):
        request=json.loads((folder/'request.json').read_text())['request']
        response=json.loads((folder/'response.json').read_text()) if (folder/'response.json').is_file() else None
        phases[request['phase']]+=1
        if response is None:errors['no_response_file']+=1
        elif response['error_type']:errors[response['error_type']]+=1
        queries.append({'call':request['call'],'tick':request['tick'],'phase':request['phase'],
                        'error_type':None if response is None else response['error_type'],'state_after':None if response is None else response['state']})
    judge=lines(run/'perception'/'judge_events.jsonl');verdicts=Counter()
    for event in judge:
        if 'error_type' in event:verdicts['transport_error']+=1;continue
        if event['parse_error']:verdicts['malformed']+=1;continue
        verdicts[json.loads(event['raw_model_response'])['verdict']]+=1
    events=lines(run/'drop'/'events.jsonl')
    states={e['tick']:(e.get('executed') or e.get('proposed_override'))['state'] for e in events if e.get('executed') or e.get('proposed_override')}
    transitions=Counter();previous='POLICY'
    for tick in sorted(states):
        if states[tick]!=previous:transitions[f'{previous}->{states[tick]}']+=1;previous=states[tick]
    executed={row['step']-1:float(row['command']) for row in lines(run/'gate_trace.jsonl')}
    policy={row['step']-1:float(row['policy_command']) for row in lines(run/'gate_trace.jsonl')}
    expected={'HOLD':1.0,'OPEN':0.0,'DONE':0.0}
    violations=[t for t,s in states.items() if s in expected and t in executed and executed[t]!=expected[s]]
    closed=[t for t in sorted(executed) if executed[t]>.5]
    first_open=next((t for t in sorted(executed) if closed and t>closed[0] and executed[t]<=.5),None)
    return {'queries':len(queries),'phases':dict(phases),'response_errors':dict(errors),
            'timeouts':errors.get('TimeoutError',0),'schema_errors':errors.get('SchemaError',0),
            'model_verdicts':dict(verdicts),'vetoed_ready':sum(bool(e.get('vetoed')) for e in judge),
            'state_transitions':dict(transitions),'hold_entries':transitions.get('POLICY->HOLD',0),
            'failed_confirmations':transitions.get('HOLD->POLICY',0),'confirmed_opens':transitions.get('HOLD->OPEN',0),
            'fallback_stops':[e for e in events if 'invalid' in e or (e.get('command') or {}).get('stop')],
            'executed_gripper':{'first_closed_tick':closed[0] if closed else None,'first_executed_open_after_close':first_open,
                                'ticks_overridden':sum(1 for t in executed if t in policy and executed[t]!=policy[t]),
                                'state_command_violations':violations},
            'query_rows':queries,
            'notes':['Outcome is scored separately by the frozen v5.1 scorer on the executed command.',
                     'Failed confirmations are HOLD->POLICY transitions, including hold timeouts.',
                     'A query with no response file means the run stopped mid-call; it is counted, not dropped.']}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    result=score(args.run);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='query_rows'},indent=2))
