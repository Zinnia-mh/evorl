"""Return-blind selection over a preregistered, finite sampler grid."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import yaml

FIELDS = {'candidate_id','ess_fraction','max_source_probability','uniform_floor',
          'fresh_coverage','expired_coverage','repeat_rate','numerical_ok','memory_mib','overhead_ms'}
TIE_BREAK = ['ess_distance','max_source_probability','repeat_rate','overhead_ms','alpha,tau,W,M_min']
ARTIFACT_ROOT=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def registration():
    candidates = [dict(alpha=a, tau=t, W=w, M_min=m) for a,t,w,m in itertools.product(
        [.25,.5], [.5,1.,2.], [3,10,30], [4,32])]
    result = dict(version='return-blind-v1', candidates=candidates, ess_target=.75,
                  thresholds=dict(max_source_probability=.5, repeat_rate=.9,
                                  memory_mib=128., overhead_ms=20., minimum_uniform_floor=.25),
                  tie_break=TIE_BREAK)
    result['sha256'] = digest(result)
    return result


def select(reg, rows):
    body = {k:v for k,v in reg.items() if k != 'sha256'}
    if digest(body) != reg.get('sha256') or reg['tie_break'] != TIE_BREAK:
        raise ValueError('registration checksum or selection rule mismatch')
    if reg != registration(): raise ValueError('registered candidate policy changed')
    if len(rows) != len(reg['candidates']):
        raise ValueError('one complete aggregate record per registered candidate required')
    trace, survivors, seen = [], [], set()
    for row in rows:
        if set(row) != FIELDS:
            raise ValueError('unknown, missing, or outcome fields in calibration input')
        i = row['candidate_id']
        if type(i) is not int or i in seen or not 0 <= i < len(reg['candidates']):
            raise ValueError('invalid or duplicate candidate identity')
        seen.add(i)
        if type(row['numerical_ok']) is not bool:
            raise ValueError('numerical_ok must be boolean')
        numeric = [float(row[k]) for k in FIELDS - {'candidate_id','numerical_ok'}]
        correct = row['numerical_ok'] and all(math.isfinite(v) and v >= 0 for v in numeric)
        correct = correct and all(0 <= float(row[k]) <= 1 for k in ['ess_fraction','max_source_probability','uniform_floor','fresh_coverage','expired_coverage','repeat_rate'])
        correct = correct and abs(row['fresh_coverage'] + row['expired_coverage'] - 1) <= 1e-6
        correct = correct and abs(row['uniform_floor']-(1-reg['candidates'][i]['alpha']))<=1e-6
        t = reg['thresholds']
        operational = correct and all(row[k] <= t[k] for k in ['max_source_probability','repeat_rate','memory_mib','overhead_ms']) and row['uniform_floor'] >= t['minimum_uniform_floor']
        trace.append(dict(candidate_id=i, correctness=bool(correct), operational=bool(operational)))
        if operational:
            c = reg['candidates'][i]
            key = (abs(row['ess_fraction']-reg['ess_target']),row['max_source_probability'],row['repeat_rate'],row['overhead_ms'],c['alpha'],c['tau'],c['W'],c['M_min'])
            survivors.append((key,i))
    if not survivors:
        raise ValueError('HOLD: no registered candidate survives')
    return dict(selected_id=min(survivors)[1], trace=sorted(trace,key=lambda x:x['candidate_id']),
                registration_sha256=reg['sha256'], thresholds=reg['thresholds'], tie_break=TIE_BREAK)


def freeze(reg, rows, output, provenance):
    if set(provenance)-{'source','input_sha256','seed','formal_seeds','commit','protocol_sha256','artifact_root'}:
        raise ValueError('unknown or outcome provenance fields')
    if not provenance.get('source') or len(provenance.get('input_sha256','')) != 64:
        raise ValueError('missing corpus provenance')
    if provenance['source']=='option-b':
        if not Path(output).resolve().is_relative_to(ARTIFACT_ROOT):
            raise ValueError('calibration artifacts must remain outside Git under registered root')
        seeds=provenance.get('formal_seeds',[])
        if provenance.get('seed')!=104729 or len(seeds)!=3 or len(set(seeds))!=3 or 104729 in seeds:
            raise ValueError('missing or conflicting formal seed registry')
        if len(provenance.get('commit',''))!=40 or len(provenance.get('protocol_sha256',''))!=64 or not provenance.get('artifact_root'):
            raise ValueError('missing registered code/protocol/corpus provenance')
    elif provenance['source']!='synthetic-test':
        raise ValueError('unsupported calibration corpus')
    elif str(Path(output).resolve()).startswith('/disk1/zmh_projects/evorl_artifacts/'):
        raise ValueError('synthetic tests cannot publish experiment artifacts')
    selection = select(reg, rows)
    c = reg['candidates'][selection['selected_id']]
    common = dict(alpha=c['alpha'], replay_temperature=c['tau'],freshness_window=c['W'],min_resident=c['M_min'])
    arms = {label:dict(common, arm=kind, score_kind=kind, diagnostic_enabled=label!='A')
            for label,kind in zip('ABCD',['uniform','fitness','td','utility'])}
    arms['A']['alpha'] = 0.
    result = dict(selection, arms=arms, provenance=provenance,
                  status='test-only' if provenance['source']=='synthetic-test' else 'selected')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'candidates.json').write_text(json.dumps(reg,sort_keys=True,indent=2)+'\n')
    with (output/'calibration_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(FIELDS)); writer.writeheader(); writer.writerows(sorted(rows,key=lambda x:x['candidate_id']))
    (output/'selection_trace.json').write_text(json.dumps(selection,sort_keys=True,indent=2)+'\n')
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(result,sort_keys=True))
    files=sorted(output.iterdir())
    (output/'checksums.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
    return result


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('register'); r.add_argument('--output',required=True)
    s=sub.add_parser('select'); s.add_argument('--registration',required=True); s.add_argument('--metrics',required=True); s.add_argument('--provenance',required=True); s.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='register':
        with Path(args.output).open('x') as f: json.dump(registration(),f,sort_keys=True,indent=2)
        return
    reg=json.loads(Path(args.registration).read_text())
    with Path(args.metrics).open(newline='') as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or []) != FIELDS: raise ValueError('forbidden or missing metric columns')
        rows=[]
        for row in reader:
            if row['numerical_ok'] not in ['true','false']: raise ValueError('invalid boolean')
            rows.append({k:(int(v) if k=='candidate_id' else v=='true' if k=='numerical_ok' else float(v)) for k,v in row.items()})
    provenance=json.loads(Path(args.provenance).read_text())
    if provenance.get('source') != 'option-b' or provenance.get('seed') != 104729:
        raise ValueError('dedicated registered Option B corpus required')
    if provenance.get('input_sha256') != hashlib.sha256(Path(args.metrics).read_bytes()).hexdigest():
        raise ValueError('input checksum mismatch')
    freeze(reg, rows, args.output, provenance)


if __name__=='__main__': main()
