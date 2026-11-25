"""Frozen CP50/CP300 D0 ranking repeatability; no environment or learner updates."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from scipy.stats import spearmanr, kendalltau
from evorl.agent import AgentState
from evorl.algorithms.td3 import TD3NetworkParams, make_mlp_td3_agent
from evorl.algorithms.erl.erl_td3.erl_learner_utility import ERLLearnerUtilityWorkflow
from evorl.algorithms.erl.erl_td3.source_replay import (
    SourceScoreTable, rank_normalize_within_checkpoint, build_weighted_source_distribution)
from evorl.envs import Box
from evorl.replay_buffers.replay_buffer import ReplayBuffer, ReplayBufferState
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.jax_utils import tree_get
from scripts.validate_replay_sampling_fidelity import path_value

ROOT = Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/replication/swimmer/seed_2047/ad300-head43221c7-v1")
SEED = 20261125


def factor_keys(seed, generation, mode, repeat, token):
    """Return source-index/reference-index/loss keys, one varying factor at a time."""
    if mode not in "ABC" or len(mode) != 1:
        raise ValueError("mode must be A, B or C")
    root = jax.random.fold_in(jax.random.PRNGKey(seed), generation)
    roots = jax.random.split(root, 3)
    return jnp.stack([
        jax.random.fold_in(jax.random.fold_in(roots[0], token), repeat if mode == "C" else 0),
        jax.random.fold_in(roots[1], repeat if mode == "A" else 0),
        jax.random.fold_in(jax.random.fold_in(roots[2], token), repeat if mode == "B" else 0),
    ])


def cohort_ranks(raw, cohorts):
    raw = np.asarray(raw)
    result = np.empty(raw.shape, dtype=np.float32)
    for c in np.unique(cohorts):
        mask = cohorts == c
        result[mask] = np.asarray(rank_normalize_within_checkpoint(
            jnp.asarray(raw[mask]), jnp.asarray(np.isfinite(raw[mask]))))
    return result


def probability_projection(table, selected_slots, ranks, *, alpha=.5, temperature=.5):
    """Use production source law; retain all resident counts and eligibility.

    One representative token per compact row is sufficient: only source law is
    consumed. No compact transition CDF or sampled index is ever used.
    """
    updated = table._replace(rank_normalized_score=table.rank_normalized_score.at[
        jnp.asarray(selected_slots)].set(jnp.asarray(ranks)))
    distribution = build_weighted_source_distribution(
        updated, updated.source_token, updated.resident_count > 0, alpha, temperature)
    p = np.asarray(distribution.source_probability, dtype=np.float64)
    return p / p.sum()


def make_measure_kernel(agent, learner_state, optimizer, critic_opt_state):
    """Compile unchanged D0 measurement, including its exact loss-key splits."""
    context = SimpleNamespace(agent=agent, optimizer=optimizer,
        _diagnostic_context=SimpleNamespace(learner_agent_state=learner_state,
                                           critic_opt_state=critic_opt_state))
    return jax.jit(lambda source, reference, key:
        ERLLearnerUtilityWorkflow._evaluate_source_utility(context, source, reference, key))


def _finite_mean(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def rank_reliability(values):
    """Descriptive pairwise reliability, with undefined correlations explicit."""
    u = np.asarray(values, dtype=np.float64)
    if u.ndim != 2 or min(u.shape) < 2 or not np.isfinite(u).all():
        raise ValueError("need finite [repeat>=2,source>=2] measurements")
    nr, ns = u.shape
    k = max(1, int(np.ceil(ns * .25)))
    sets, boundary_ties = [], 0
    for row in u:
        order = np.argsort(-row, kind="stable")
        sets.append(set(order[:k]))
        boundary_ties += int(k < ns and row[order[k-1]] == row[order[k]])
    tri = np.triu_indices(ns, 1)
    directions = np.sign(u[:, tri[0]] - u[:, tri[1]])
    correlations, taus, overlaps, flips, strict_fractions, signs = [], [], [], [], [], []
    for a, b in itertools.combinations(range(nr), 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlations.append(float(spearmanr(u[a], u[b]).statistic))
            taus.append(float(kendalltau(u[a], u[b], variant="b").statistic))
        overlaps.append(len(sets[a] & sets[b]) / k)
        strict = (directions[a] != 0) & (directions[b] != 0)
        strict_fractions.append(float(strict.mean()))
        flips.append(float((directions[a][strict] != directions[b][strict]).mean())
                     if strict.any() else None)
        signs.append(float((np.sign(u[a]) == np.sign(u[b])).mean()))
    modal_sign = np.maximum.reduce([(u > 0).sum(0), (u < 0).sum(0), (u == 0).sum(0)]) / nr
    variances = np.var(u, axis=0, ddof=1)
    return dict(source_count=ns, repeat_count=nr, repeat_pairs=len(overlaps),
        utility_mean=float(u.mean()), utility_mean_median=float(np.median(u.mean(0))),
        within_source_variance_mean=float(variances.mean()),
        within_source_variance_median=float(np.median(variances)),
        between_source_mean_variance=float(np.var(u.mean(0), ddof=1)),
        positive_fraction=float((u > 0).mean()),
        sign_agreement_mean=float(np.mean(signs)),
        modal_sign_fraction_mean=float(modal_sign.mean()),
        always_same_sign_source_fraction=float((modal_sign == 1).mean()),
        spearman_mean=_finite_mean(correlations), kendall_mean=_finite_mean(taus),
        spearman_defined_pairs=int(np.isfinite(correlations).sum()),
        kendall_defined_pairs=int(np.isfinite(taus).sum()),
        spearman_min=float(min(correlations)) if all(np.isfinite(correlations)) else None,
        spearman_max=float(max(correlations)) if all(np.isfinite(correlations)) else None,
        top25_overlap_mean=float(np.mean(overlaps)),
        top25_boundary_tie_repeats=boundary_ties,
        pair_reversal_mean=_finite_mean(flips),
        strict_pair_fraction=float(np.mean(strict_fractions)),
        pairwise_spearman=[v if np.isfinite(v) else None for v in correlations],
        pairwise_kendall=[v if np.isfinite(v) else None for v in taus],
        pairwise_top25_overlap=overlaps, pairwise_reversal=flips)


def full_state_hash(raw, metadata):
    digest = hashlib.sha256()
    for spec in metadata["leaf_specs"]:
        v = np.asarray(path_value(raw, spec["path"]))
        if list(v.shape) != spec["shape"] or str(v.dtype) != spec["dtype"]:
            raise ValueError("native checkpoint leaf shape/dtype mismatch")
        digest.update(spec["path"].encode())
        digest.update(str((v.dtype, v.shape)).encode())
        digest.update(v.tobytes())
    return digest.hexdigest()


def restore_frozen(root, generation):
    candidates = list((root / "runs/D0/segments").glob(f"*/checkpoints/{generation}"))
    if len(candidates) != 1:
        raise ValueError("historical checkpoint must be unambiguous")
    cp = candidates[0]
    metadata = json.loads((cp / "runtime/metadata").read_text())
    with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
        raw = checkpointer.restore(str(cp / "state"), args=ocp.args.StandardRestore(
            fallback_sharding=jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])))
    if full_state_hash(raw, metadata) != metadata["state_sha256"]:
        raise ValueError("native checkpoint full-state checksum mismatch")
    if int(raw[3]["iterations"]) != generation:
        raise ValueError("checkpoint generation mismatch")
    return raw, metadata, cp


def frozen_components(raw, config):
    if config.normalize_obs or config.optimizer.grad_clip_norm != 0 or config.num_rl_agents != 1:
        raise ValueError("protocol needs unnormalized single learner and plain Adam")
    first = lambda x: x[0]
    params = TD3NetworkParams(**jax.tree.map(first, raw[0]["params"]))
    state = AgentState(params=params)
    optimizer = optax.adam(config.optimizer.lr)
    adam = raw[4][1][0]
    opt_state = (optax.ScaleByAdamState(count=adam["count"],
        mu=jax.tree.map(first, adam["mu"]), nu=jax.tree.map(first, adam["nu"])), optax.EmptyState())
    expected = optimizer.init(params.critic_params)
    if jax.tree.structure(expected) != jax.tree.structure(opt_state):
        raise ValueError("restored Critic optimizer structure mismatch")
    for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(opt_state)):
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("restored Critic optimizer leaf mismatch")
    d = raw[5]["data"]
    if d["dones"] is not None or d["next_obs"] is not None or d["extras"][1] != {}:
        raise ValueError("unexpected native Replay payload schema")
    data = SampleBatch(obs=d["obs"], actions=d["actions"], rewards=d["rewards"],
        extras=PyTreeDict(env_extras=PyTreeDict(
            ori_obs=d["extras"][0][0], termination=d["extras"][0][1]),
            policy_extras=PyTreeDict()))
    action_size = d["actions"].shape[-1]
    agent = make_mlp_td3_agent(Box(low=-jnp.ones(action_size), high=jnp.ones(action_size)),
        **dict(config.agent_network), discount=config.discount,
        exploration_epsilon=config.exploration_epsilon, policy_noise=config.policy_noise,
        clip_policy_noise=config.clip_policy_noise, critics_in_actor_loss=config.critics_in_actor_loss,
        normalize_obs=config.normalize_obs)
    state, opt_state, data = jax.device_put((state, opt_state, data), jax.devices("gpu")[0])
    rb_state = ReplayBufferState(data=data, buffer_size=jnp.asarray(raw[5]["buffer_size"]),
        current_index=jnp.asarray(raw[5]["current_index"]))
    return agent, state, optimizer, opt_state, rb_state


def tv_summary(probabilities, baseline):
    pairs = [float(np.abs(a-b).sum()/2)
             for a,b in itertools.combinations(probabilities,2)]
    return dict(pairwise_tv_mean=float(np.mean(pairs)), pairwise_tv_max=float(np.max(pairs)),
        tv_uniform_mean=float(np.mean(np.abs(probabilities-baseline).sum(-1)/2)),
        pairwise_tv=pairs)


def analyze_mode(utilities, cohorts, table, slots, generation, mode):
    ranks = np.stack([cohort_ranks(u, cohorts) for u in utilities])
    global_ranks = np.stack([np.asarray(rank_normalize_within_checkpoint(
        jnp.asarray(u), jnp.ones(u.shape, bool))) for u in utilities])
    rng = np.random.default_rng(np.random.SeedSequence([SEED,generation,ord(mode),991]))
    permuted = ranks.copy()
    for row in permuted:
        for c in np.unique(cohorts):
            mask = cohorts == c
            row[mask] = rng.permutation(row[mask])
    p = np.stack([probability_projection(table,slots,row) for row in ranks])
    null_p = np.stack([probability_projection(table,slots,row) for row in permuted])
    global_p = np.stack([probability_projection(table,slots,row) for row in global_ranks])
    counts = np.asarray(table.resident_count,dtype=float)
    baseline = counts/counts.sum()
    cohort_stats = {str(c):rank_reliability(utilities[:,cohorts==c]) for c in np.unique(cohorts)}
    stats = dict(global_raw=rank_reliability(utilities), cohort=cohort_stats,
        cohort_spearman_mean=_finite_mean([s["spearman_mean"] for s in cohort_stats.values()]),
        cohort_kendall_mean=_finite_mean([s["kendall_mean"] for s in cohort_stats.values()]),
        cohort_top25_overlap_mean=_finite_mean([s["top25_overlap_mean"] for s in cohort_stats.values()]),
        cohort_pair_reversal_mean=_finite_mean([s["pair_reversal_mean"] for s in cohort_stats.values()]),
        normalized_cohort_rank_reliability=rank_reliability(ranks),
        permutation_rank_reliability=rank_reliability(permuted),
        cohort_vs_global_spearman_mean=float(np.mean([
            spearmanr(a,b).statistic for a,b in zip(ranks,global_ranks)])),
        sampling=tv_summary(p,baseline), permutation_sampling=tv_summary(null_p,baseline),
        cohort_vs_global_probability_tv_mean=float(np.abs(p-global_p).sum(-1).mean()/2))
    return stats, ranks, p, null_p


def run_checkpoint(root, out, generation, config, repeats):
    raw, metadata, cp = restore_frozen(root,generation)
    table_raw = raw[6]["score_table"]
    resident = (np.asarray(table_raw["source_token"]) >= 0) & (np.asarray(table_raw["resident_count"]) > 0)
    row_indices = np.flatnonzero(resident)
    table = SourceScoreTable(**{name:jnp.asarray(np.asarray(value)[resident] if np.ndim(value) else value)
        for name,value in table_raw.items()})
    slots = np.flatnonzero(np.asarray(table.weighted_eligible))
    slots = slots[np.argsort(np.asarray(table.source_token)[slots])]
    tokens = np.asarray(table.source_token)[slots]
    cohorts = np.asarray(table.measured_generation)[slots]
    if len(tokens)!=128 or any((cohorts==c).sum()!=32 for c in np.unique(cohorts)):
        raise ValueError("protocol expects128 sources in four complete32-source cohorts")
    rb = raw[5]
    origins, rb_tokens = np.asarray(rb["origin_type"]), np.asarray(rb["source_token"])
    generations = np.asarray(rb["source_generation"])
    if generations[origins==0].max()>generation:
        raise ValueError("future source in restored Replay")
    resident_indices = [np.flatnonzero((origins==0)&(rb_tokens==t)) for t in tokens]
    for slot, indices in zip(slots,resident_indices):
        if len(indices)!=int(table.resident_count[slot]) or len(indices)<config.source_batch_size:
            raise ValueError("source residency mismatch or insufficient minibatch")
    selection = [dict(token=int(t),cohort=int(c),resident_count=len(i),registry_slot=int(row_indices[s]))
        for t,c,i,s in zip(tokens,cohorts,resident_indices,slots)]
    directory = out/f"cp{generation}"
    directory.mkdir(exist_ok=False)
    (directory/"selection.json").write_text(json.dumps(dict(
        checkpoint=str(cp),state_sha256=metadata["state_sha256"],sources=selection,
        rule="all weighted-eligible sources; token-sorted; no score/Return selection"),indent=2))
    original = probability_projection(table, slots, np.asarray(table.rank_normalized_score)[slots])
    saved = np.asarray(raw[6]["distribution"]["source_probability"],dtype=float)[row_indices]
    saved /= saved.sum()
    if not np.allclose(original,saved,rtol=1e-6,atol=1e-8):
        raise ValueError("production projection differs from frozen source law")
    agent, state, optimizer, opt, rb_state = frozen_components(raw,config)
    kernel = make_measure_kernel(agent,state,optimizer,opt)
    buffer = ReplayBuffer(capacity=int(config.replay_buffer_capacity),sample_batch_size=int(config.batch_size))
    n, bs = len(tokens), int(config.source_batch_size)
    source_indices = np.empty((3,repeats,n,bs),np.int32)
    reference_indices = np.empty((3,repeats,int(config.batch_size)),np.int32)
    keys = np.empty((3,repeats,n,3,2),np.uint32)
    utilities, loss_before, loss_after, td_error = [np.empty((3,repeats,n),np.float32) for _ in range(4)]
    timings = {}
    k = factor_keys(SEED,generation,"A",0,int(tokens[0]))
    source = tree_get(rb_state.data,jnp.asarray(resident_indices[0][:bs]))
    reference = buffer.sample(rb_state,k[1])
    start = time.perf_counter()
    jax.block_until_ready(kernel(source,reference,k[2]))
    timings["compile_warmup_seconds"] = time.perf_counter()-start
    summary = {}
    rank_arrays, probabilities, null_probabilities = [],[],[]
    for mi,mode in enumerate("ABC"):
        start=time.perf_counter()
        for repeat in range(repeats):
            reference_key = factor_keys(SEED,generation,mode,repeat,int(tokens[0]))[1]
            ref_indices = jax.random.randint(reference_key,(int(config.batch_size),),
                minval=0,maxval=rb_state.buffer_size)
            reference_indices[mi,repeat]=np.asarray(ref_indices)
            reference = buffer.sample(rb_state,reference_key)
            for si,(token,indices) in enumerate(zip(tokens,resident_indices)):
                k=factor_keys(SEED,generation,mode,repeat,int(token))
                keys[mi,repeat,si]=np.asarray(k)
                chosen=jax.random.choice(k[0],jnp.asarray(indices),(bs,),replace=False)
                source_indices[mi,repeat,si]=np.asarray(chosen)
                source=tree_get(rb_state.data,chosen)
                td,result=kernel(source,reference,k[2])
                utilities[mi,repeat,si]=float(result.utility)
                loss_before[mi,repeat,si]=float(result.loss_ref_before)
                loss_after[mi,repeat,si]=float(result.loss_ref_after)
                td_error[mi,repeat,si]=float(td.mean)
            print(f"CP{generation} mode={mode} repeat={repeat+1}/{repeats} finite={np.isfinite(utilities[mi,repeat]).all()}",flush=True)
        timings[f"{mode}_measurement_seconds"]=time.perf_counter()-start
        summary[mode], ranks, p, null_p = analyze_mode(utilities[mi],cohorts,table,slots,generation,mode)
        rank_arrays.append(ranks);probabilities.append(p);null_probabilities.append(null_p)
    if not all(np.isfinite(x).all() for x in (utilities,loss_before,loss_after,td_error)):
        raise ValueError("nonfinite real measurements")
    for mode,mi,varying in [("A",0,1),("B",1,2),("C",2,0)]:
        for ki in range(3):
            if ki!=varying and not np.all(keys[mi,:,:,ki,:]==keys[mi,0,:,ki,:]):
                raise ValueError("unintended key factor changed")
        if mode!="C" and not np.all(source_indices[mi]==source_indices[mi,0]):
            raise ValueError("fixed source minibatch changed")
        if mode!="A" and not np.all(reference_indices[mi]==reference_indices[mi,0]):
            raise ValueError("fixed reference minibatch changed")
    if not np.array_equal(utilities[:,0],np.broadcast_to(utilities[0,0],utilities[:,0].shape)):
        raise ValueError("shared anchor measurements differ across factor arms")
    if len({x.tobytes() for x in reference_indices[0]})!=repeats:
        raise ValueError("A reference draws are not independent")
    after_hash=full_state_hash(raw,metadata)
    if after_hash!=metadata["state_sha256"] or jax.config.jax_enable_x64:
        raise ValueError("frozen state or global precision changed")
    np.savez_compressed(directory/"measurements.npz",utility=utilities,loss_before=loss_before,
        loss_after=loss_after,td_error=td_error,source_indices=source_indices,
        reference_indices=reference_indices,keys=keys,tokens=tokens,cohorts=cohorts,
        cohort_ranks=np.stack(rank_arrays),source_probability=np.stack(probabilities),
        permutation_probability=np.stack(null_probabilities),
        resident_tokens=np.asarray(table.source_token),resident_counts=np.asarray(table.resident_count))
    result=dict(generation=generation,checkpoint=str(cp),source_count=n,repeats=repeats,
        real_utility_measurement_count=3*repeats*n,unique_factor_schedules=(1+3*(repeats-1))*n,
        step_scale=1.,source_batch_size=bs,reference_batch_size=int(config.batch_size),
        before_state_sha256=metadata["state_sha256"],after_state_sha256=after_hash,
        frozen_state_verified=True,original_probability_law_verified=True,
        factor_isolation_verified=True,global_x64=False,timing=timings,modes=summary)
    (directory/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=8)
    args=p.parse_args()
    out,root=args.output.resolve(),args.input_root.resolve()
    if out.is_relative_to(root) or args.repeats!=8 or jax.default_backend()!="gpu":
        raise ValueError("need new external output, preregistered8 repeats and real GPU")
    out.mkdir(exist_ok=False)
    config=OmegaConf.load(root/"runs/D0/resolved_config.yaml")
    assert (config.alpha,config.replay_temperature,config.diagnostic_interval,config.freshness_window)==(.5,.5,10,30)
    here=Path(__file__).resolve()
    manifest=dict(base_commit="404a01b3d796967656d8b9309d3c460a4c24301d",
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        code_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((root/"runs/D0/resolved_config.yaml").read_bytes()).hexdigest(),
        seed=SEED,training_seed=2047,checkpoints=[50,300],repeats=8,source_selection="all128",
        step_scale=1.,environment_interactions=0,training_updates=0,
        backend=jax.default_backend(),device=str(jax.devices()[0]),
        protocol="A reference only; B TD3 loss target keys only; C source minibatch only",
        note="repeat0 anchor shared; B/C fixed-reference conditional tests; one training seed")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    started=time.perf_counter()
    results={g:run_checkpoint(root,out,g,config,args.repeats) for g in (50,300)}
    (out/"summary.json").write_text(json.dumps(dict(
        wall_seconds=time.perf_counter()-started,checkpoints=results),indent=2,allow_nan=False))
    print("PASS frozen real TD3 Utility repeatability diagnostic",flush=True)


if __name__=="__main__":
    main()
