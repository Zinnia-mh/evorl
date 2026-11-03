"""One native million-capacity full-state I/O probe; NO rollout or training."""
import argparse,json,os,subprocess,time
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import PilotCheckpointStore,restore_template,mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_smoke_workflow import initialization_fingerprint
from evorl.replay_buffers.source_aware_replay_buffer import encode_source_token

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
    directory=Path(args.directory).resolve()
    allowed=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/checkpoint_validation')
    if not directory.is_relative_to(allowed) or directory.exists():raise ValueError('exclusive external validation directory required')
    if jax.default_backend()!='gpu':raise ValueError('this one-off transfer/I/O probe requires the original GPU backend')
    frozen=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-selected-v2-head-dbcb4f9-seed104729')
    c=OmegaConf.load(frozen/'arm_A.yaml');assert int(c.replay_buffer_capacity)==1000000
    w=type('CheckpointIOProbe',(ERLUtilityReplayWorkflow,),{}).build_from_config(c,enable_jit=True)
    # Native schema + real initialized networks/optimizers/population. Payload
    # is explicitly synthetic storage-test data, never a formal training result.
    state=restore_template(w,abstract=False);replay=state.replay_buffer_state;capacity=replay.source_token.size
    leaves,tree=jax.tree_util.tree_flatten(replay.data);keys=jax.random.split(jax.random.PRNGKey(130363),len(leaves))
    values=[jax.random.normal(k,x.shape,dtype=x.dtype) for k,x in zip(keys,leaves)]
    data=jax.tree_util.tree_unflatten(tree,values)
    index=jnp.arange(capacity,dtype=jnp.int32);ea=index%5!=0;token_base=encode_source_token(1,0,int(c.pop_size))
    replay=replay.replace(data=data,buffer_size=jnp.asarray(capacity,replay.buffer_size.dtype),current_index=jnp.asarray(0,replay.current_index.dtype),origin_type=jnp.where(ea,0,1).astype(replay.origin_type.dtype),source_token=jnp.where(ea,token_base+index%int(c.pop_size),-1).astype(replay.source_token.dtype),source_generation=jnp.where(ea,1,0).astype(replay.source_generation.dtype),score_version=jnp.full_like(replay.score_version,-1))
    state=state.replace(replay_buffer_state=replay);jax.block_until_ready(state)
    logical=sum(int(x.size)*np.dtype(x.dtype).itemsize for x in jax.tree.leaves(state))
    directory.mkdir(parents=True,exist_ok=False);fs=os.statvfs(directory);free=fs.f_bavail*fs.f_frsize
    if free<4*logical:raise ValueError('insufficient safe checkpoint disk headroom')
    runtime=dict(purpose='storage validation only',synthetic_transition_payload=True,environment_interactions=0,optimizer_updates=0)
    start=time.perf_counter()
    with PilotCheckpointStore(directory/'checkpoints',c) as store:path=store.save(state,runtime,force=True)
    save_seconds=time.perf_counter()-start
    template=restore_template(w);start=time.perf_counter()
    with PilotCheckpointStore(directory/'checkpoints',c,read_only=True) as store:restored,metadata=store.restore(0,template)
    jax.block_until_ready(restored);load_seconds=time.perf_counter()-start
    assert metadata==runtime and int(restored.replay_buffer_state.buffer_size)==capacity
    assert initialization_fingerprint(restored)==initialization_fingerprint(state)
    size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    result=dict(status='PASS',backend=jax.default_backend(),capacity=int(capacity),resident_transitions=int(restored.replay_buffer_state.buffer_size),native_checkpoint_bytes=size,logical_state_bytes=logical,write_seconds=save_seconds,load_seconds=load_seconds,disk_free_before_bytes=free,recommended_active_segment_headroom_bytes=4*size,scope='one save and one restore, GPU transfer + hashes + native synchronous commit included; warm OS-cache load; workflow construction/template tracing excluded',synthetic_transition_payload=True,environment_interactions=0,optimizer_updates=0,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_mechanism_digest=mechanism_digest(),checkpoint=str(path))
    with (directory/'io_report.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
    print(json.dumps(result),flush=True);w.close()

if __name__=='__main__':main()
