from pathlib import Path
import json,numpy as np
def main():
    r=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/rl-guided-population-seed3083-v1')
    paths=sorted((r/'runs/M').glob('segments/*/evolution.jsonl'),key=lambda p:p.stat().st_mtime)
    u={x['generation']:x for p in paths for x in map(json.loads,p.read_text().splitlines())}
    rows=[u[g] for g in sorted(u) if not u[g].get('bootstrap')]
    lifetimes={g:0 for g in range(60,151,10)}
    for row in rows:
     for line in row['evaluated_lineage']:
      for root in line['roots']:lifetimes[root//100]=max(lifetimes.get(root//100,0),row['generation']-root//100)
    stats={}
    for g,row in u.items():
     prev=u.get(g-1)
     if not prev or 'fitness' not in prev:continue
     for i,childfitness in enumerate(row['fitness']):
      if row['injected'] and i>=24:continue
      parent=prev['parent_a'][i]
      if parent<0:continue
      kind=('M' if prev['evaluated_lineage'][parent]['roots'] else 'nonM')+('_elite' if i<4 else '_offspring')
      stats.setdefault(kind,[]).append(float(childfitness)-prev['fitness'][parent])
    result=dict(root_cohort_max_observed_age=lifetimes,
     mean_evaluated_M_slots=float(np.mean([x['derived_count'] for x in rows])),
     total_M_source_slots=sum(x['derived_count'] for x in rows),
     total_M_descendant_slots=sum(sum(bool(l['roots']) and l['depth']>0 for l in row['evaluated_lineage']) for row in rows),
     min_no_M_evaluated=min(32-x['derived_count'] for x in rows),
     parent_child_fitness_changes={k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),positive_fraction=float(np.mean(np.array(v)>0))) for k,v in stats.items()},
     injection_summary=[])
    for x in rows:
     if x['injected']:
      direct=x['fitness'][24:];other=x['fitness'][:24]
      result['injection_summary'].append(dict(generation=x['generation'],
       mean_direct_fitness=float(np.mean(direct)),mean_other24_fitness=float(np.mean(other)),
       next_generation_M_ancestry=x['derived_next_count'],next_generation_current_cohort=sum(any(root//100==x['generation'] for root in item['roots']) for item in x['next_lineage']),M_count=x['derived_count'],
       before_pair_distance=x['before_injection_geometry']['all']['mean_pairwise_distance'],
       after_pair_distance=x['geometry']['all']['mean_pairwise_distance'],
       M_group_distance_to_RL=x['geometry']['M_ancestry']['mean_distance_to_RL']))
    (r/'mechanism_details.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    m=json.loads((r/'summary-M.json').read_text())
    print('M physical wall,core total',m['process_wall_seconds'],m['core_training_seconds'])

if __name__=='__main__':main()
