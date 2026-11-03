"""Pilot full-state checkpoints using native Orbax atomic Composite commits."""
import hashlib,importlib.metadata,json,platform
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from evorl.types import State
from evorl.utils.orbax_utils import filter_zero_size_arrays_on_save
from .erl_utility_replay import init_utility_replay_state
from .utility_replay_smoke_workflow import initialization_fingerprint

def config_contract(config):
    data=OmegaConf.to_container(config,resolve=True)
    for name in ['output_dir','arm_manifest_path','checkpoint','recorders','project','tags']:data.pop(name,None)
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def mechanism_digest():
    root=Path(__file__).resolve().parents[4];digest=hashlib.sha256()
    paths=sorted((root/'evorl').rglob('*.py'))
    runner=root/'scripts/run_utility_replay_training.py'
    if runner.exists():paths.append(runner)
    for path in paths:digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()

def environment_contract():
    names=['jax','jaxlib','orbax-checkpoint','numpy','brax','flax','optax']
    return dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in names},backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),prng=str(jax.config.jax_default_prng_impl),jit_disabled=bool(jax.config.jax_disable_jit))

def restore_template(workflow,*,abstract=True):
    """Abstract native components only: never setup, warmup, rollout or evaluate."""
    def build(root):
        key,agent_key,rb_key=jax.random.split(root,3)
        agent,opt,ec=workflow._setup_agent_and_optimizer(agent_key)
        return State(key=key,metrics=workflow._setup_workflow_metrics(),agent_state=agent,opt_state=opt,ec_opt_state=ec,replay_buffer_state=workflow._setup_replaybuffer(rb_key),utility_replay_state=init_utility_replay_state(workflow.replay_buffer.capacity,jax.random.fold_in(root,jnp.uint32(0x55525037))))
    key=jax.random.PRNGKey(int(workflow.config.seed))
    return jax.eval_shape(build,key) if abstract else build(key)

def _empty(x):return hasattr(x,'shape') and np.prod(x.shape)==0

def _leaf_specs(state):
    return [dict(path=jax.tree_util.keystr(path),shape=list(x.shape),dtype=str(x.dtype))
            for path,x in jax.tree_util.tree_flatten_with_path(state)[0]]

def _saved_template(template,specs):
    pairs,tree=jax.tree_util.tree_flatten_with_path(template)
    if len(pairs)!=len(specs):raise ValueError('checkpoint state structure mismatch')
    leaves=[]
    for (path,x),spec in zip(pairs,specs):
        name=jax.tree_util.keystr(path)
        if name!=spec['path'] or str(x.dtype)!=spec['dtype']:raise ValueError('checkpoint state path/dtype mismatch')
        shape=tuple(spec['shape'])
        if shape!=tuple(x.shape):
            # Source-distribution arrays are capacity-sized only after refresh.
            allowed=any(name.endswith('.distribution.'+field)
                        for field in ['baseline_source_probability','weighted_source_probability','source_probability'])
            if not allowed or len(shape)!=1:raise ValueError('unexpected checkpoint state shape change: '+name)
        leaves.append(jax.ShapeDtypeStruct(shape,x.dtype))
    return jax.tree_util.tree_unflatten(tree,leaves)

class PilotCheckpointStore:
    def __init__(self,directory,config,*,interval=50,keep=3,read_only=False):
        if interval<1 or keep<2:raise ValueError('checkpoint interval>=1 and keep>=2 required')
        self.directory=Path(directory).resolve();self.config=config;self.read_only=read_only
        self.seals=self.directory.parent/'checkpoint_checksums'
        if not read_only:self.seals.mkdir(parents=True,exist_ok=True)
        self.manager=ocp.CheckpointManager(str(self.directory),options=ocp.CheckpointManagerOptions(save_interval_steps=interval,max_to_keep=None if read_only else keep,read_only=read_only,create=not read_only,cleanup_tmp_directories=False,enable_async_checkpointing=False),item_handlers=dict(state=ocp.StandardCheckpointHandler(),runtime=ocp.JsonCheckpointHandler()))
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.manager.close()
    def steps(self):return self.manager.all_steps(read=True)
    def _files(self,path):return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob('*')) if p.is_file()}
    def save(self,state,runtime,*,force=False):
        if self.read_only:raise ValueError('read-only checkpoint store')
        step=int(jax.device_get(state.metrics.iterations))
        if step in self.steps():raise FileExistsError(f'checkpoint {step} already exists')
        if not force and not self.manager.should_save(step):return None
        jax.block_until_ready(state)
        payload=dict(schema=2,config_contract=config_contract(self.config),mechanism_digest=mechanism_digest(),environment=environment_contract(),state_sha256=initialization_fingerprint(state),leaf_specs=_leaf_specs(state),generation=step,runtime=runtime)
        json.dumps(payload,allow_nan=False)
        saved=self.manager.save(step,args=ocp.args.Composite(state=ocp.args.StandardSave(filter_zero_size_arrays_on_save(state)),runtime=ocp.args.JsonSave(payload)),force=force)
        self.manager.wait_until_finished();self.manager.check_for_errors()
        if not saved:raise RuntimeError('native checkpoint save was not committed')
        path=self.directory/str(step)
        with (self.seals/f'{step}.json').open('x') as f:json.dump(self._files(path),f,sort_keys=True)
        return path
    def restore(self,step,template):
        if step not in self.steps():raise FileNotFoundError(f'no committed checkpoint {step}')
        path=self.directory/str(step);seal=self.seals/f'{step}.json'
        if not seal.exists() or self._files(path)!=json.loads(seal.read_text()):raise ValueError('checkpoint checksum mismatch or missing seal')
        with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as reader:
            metadata=reader.restore(str(path/'runtime'),args=ocp.args.JsonRestore())
        if metadata.get('schema')!=2 or metadata['config_contract']!=config_contract(self.config) or metadata['mechanism_digest']!=mechanism_digest() or metadata['environment']!=environment_contract():raise ValueError('incompatible checkpoint configuration, implementation or environment')
        template=_saved_template(template,metadata['leaf_specs'])
        abstract=jax.tree.map(lambda x:None if _empty(x) else x,template)
        result=self.manager.restore(step,args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract),runtime=ocp.args.JsonRestore()))
        restored=jax.tree.map(lambda src,dst:jnp.zeros(src.shape,src.dtype) if _empty(src) else dst,template,result.state)
        if int(restored.metrics.iterations)!=step or initialization_fingerprint(restored)!=metadata['state_sha256']:raise ValueError('restored full-state checksum mismatch')
        return restored,metadata['runtime']
