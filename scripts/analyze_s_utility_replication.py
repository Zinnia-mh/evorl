"""CPU-only analysis of completed fixed-seed replication; no training imports."""
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/s-utility-replication-seed3083-v1')
def auc(evals,first=20,last=150):
    selected=[r for r in evals if first<=r['generation']<=last]
    return float(np.trapz([r['return_mean'] for r in selected],[r['environment_timesteps'] for r in selected]))
def main():
    data=json.loads((ROOT/'summary.json').read_text())
    common=json.loads((ROOT/'shared_warmup.json').read_text())
    result={}
    curve=[]
    checks={}
    for arm,d in data.items():
        p=d['performance'];ev=p['evaluations']
        assert [r['generation'] for r in ev]==list(range(20,151,10))
        assert all(math.isfinite(r['return_mean']) for r in ev)
        assert p['initial']==data['A']['performance']['initial']
        assert p['final']==data['A']['performance']['final']
        assert math.isclose(auc(ev),p['auc'],rel_tol=1e-12)
        timing={r['generation']:r for r in d['evaluation_timings']}
        for e in ev:
            assert timing[e['generation']]['return_mean']==e['return_mean']
            curve.append(dict(arm=arm,**e,training_elapsed_seconds=timing[e['generation']]['training_elapsed_seconds']))
        segs=list((ROOT/'runs'/arm/'segments').iterdir())
        rows=[json.loads(line) for seg in segs for line in (seg/'schedule.jsonl').read_text().splitlines()]
        assert len(rows)==140 and [r['generation'] for r in rows]==list(range(11,151))
        assert all(r['baseline_prng_ok'] for r in rows)
        assert all(r['new_scores']==(0 if arm=='A' else (r['generation']-10)*32//10-(r['generation']-11)*32//10) for r in rows)
        audits=[r for r in rows if r.get('real_critic_input_audit')]
        assert arm=='A' or ([r['generation'] for r in audits]==[20,150] and all(r['rl_changed']==0 and r['changed_ea']>0 for r in audits))
        assert d['discarded_attempt_score_calls']==0
        threshold=next((r for r in d['evaluation_timings'] if r['return_mean']>=350),None)
        assert threshold==d['first350']
        if arm!='A':
            assert p['restored_state_sha256']==common['full_state_sha256']
            assert d['final_prng']==data['A']['final_prng'] and d['final_ec_prng']==data['A']['final_ec_prng']
        checks[arm]=dict(real_scores=d['real_scores_total'],baseline_prng_all=True,
            actual_critic_audits=audits,restored_full_state_sha256=p.get('restored_state_sha256'),
            complete_checkpoint=p['last_checkpoint'])
        result[arm]=dict(auc20_150=p['auc'],auc50_150=auc(ev,50),auc70_150=auc(ev,70),
            last5_mean=float(np.mean([r['return_mean'] for r in ev[-5:]])),
            final_return=p['final_return'],first350=threshold,
            measured_training_seconds=d['measured_total_training_seconds'],
            postwarmup_step_seconds=d['mechanism']['step_seconds'],process_wall_seconds=d['wall_seconds'],
            coverage_mean=d['mechanism']['eligible_coverage_mean'],
            ea_tv_mean=d['mechanism']['ea_conditional_tv_mean'])
        if arm=='P':
            perm=[json.loads(line) for seg in segs for line in (seg/'permutation.jsonl').read_text().splitlines()]
            assert len(perm)==140 and all(r['preserved_rank_multisets'] for r in perm)
            checks[arm]['permutation_seconds']=sum(r['seconds'] for r in perm)
    pairs={}
    for numerator,denominator in [('S','A'),('P','A'),('S','P')]:
        pairs[numerator+'/'+denominator]={k:result[numerator][k]/result[denominator][k]-1 for k in ['auc20_150','auc50_150','auc70_150','measured_training_seconds']}
    s,p,a=[result[k]['auc20_150'] for k in ('S','P','A')]
    route=('A' if s>a and p>a and s/p-1>=.10 else
           'B' if s>a and p>a and abs(s/p-1)<.10 else
           'C' if s<=a and p<=a else 'mixed')
    final=dict(seed=3083,arms=result,pairs=pairs,descriptive_route=route,checks=checks,
        shared_warmup=common,physical_training_steps=14180000,physical_evaluation_steps=5376000)
    (ROOT/'comparison.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    with (ROOT/'learning_curves.csv').open('w',newline='') as f:
        fields=['arm','generation','environment_timesteps','return_mean','training_elapsed_seconds']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(curve)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm in ('A','S','P'):
        subset=[r for r in curve if r['arm']==arm]
        for ax,xlabel,values in [(axes[0],'Training environment steps (million)',[r['environment_timesteps']/1e6 for r in subset]),
                                (axes[1],'Measured training time (minutes)',[r['training_elapsed_seconds']/60 for r in subset])]:
            ax.plot(values,[r['return_mean'] for r in subset],marker='o',label=arm)
            ax.set_xlabel(xlabel);ax.set_ylabel('Evaluation mean Return (128 episodes)');ax.grid(alpha=.25)
            ax.axhline(350,color='gray',linestyle='--',linewidth=.8);ax.legend()
    fig.suptitle('Fixed independent seed 3083 | common CP10 to CP150')
    fig.savefig(ROOT/'learning_curves.png',dpi=180)
    fig.savefig(ROOT/'learning_curves.pdf')
    print(json.dumps({k:final[k] for k in ('seed','arms','pairs','descriptive_route')},indent=2))
if __name__=='__main__':main()
