"""Host-boundary byte evidence. No learner, sampler or state mutation."""
import dataclasses,hashlib,json
from collections.abc import Mapping
from pathlib import Path
import jax
import numpy as np
from .utility_replay_smoke_workflow import initialization_fingerprint

def _walk(value,path=''):
    if value is None:return
    if isinstance(value,Mapping):
        for name in sorted(value):yield from _walk(value[name],f'{path}.{name}' if path else str(name))
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            if not field.metadata.get('static',False):yield from _walk(getattr(value,field.name),f'{path}.{field.name}')
    elif isinstance(value,tuple) and hasattr(value,'_fields'):
        for name in value._fields:yield from _walk(getattr(value,name),f'{path}.{name}')
    elif isinstance(value,(tuple,list)):
        for i,child in enumerate(value):yield from _walk(child,f'{path}[{i}]')
    else:yield path,value

def _group(path):
    for name in ['target_actor','target_critic','actor','critic']:
        if path.startswith(f'agent_state.params.{name}_params'):return name
    if path.startswith('opt_state.actor'):return 'actor_optimizer'
    if path.startswith('opt_state.critic'):return 'critic_optimizer'
    if path.startswith('ec_opt_state.key'):return 'ea_prng'
    if path.startswith('ec_opt_state'):return 'ea_population_and_state'
    if path=='key':return 'baseline_prng'
    if path.startswith('replay_buffer_state.data'):return 'replay_payload'
    if path.startswith('replay_buffer_state.'):
        return 'replay_pointer' if path.rsplit('.',1)[-1] in ['buffer_size','current_index'] else 'replay_provenance'
    if path.startswith('utility_replay_state.intervention_key'):return 'intervention_prng'
    if path.startswith('utility_replay_state.score_table'):return 'utility_score_table'
    if path.startswith('utility_replay_state'):return 'utility_attached_state'
    return 'other_state'

def _array(leaf):
    if hasattr(leaf,'dtype') and jax.dtypes.issubdtype(leaf.dtype,jax.dtypes.prng_key):leaf=jax.random.key_data(leaf)
    return np.asarray(jax.device_get(leaf))

def _summary(array):
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest(),bytes=array.nbytes)

def capture_evidence(state,directory,*,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    legacy=initialization_fingerprint(state);jax_leaves=jax.tree_util.tree_flatten_with_path(state)[0]
    semantic=list(_walk(state));assert len(semantic)==len(jax_leaves),'unsupported state traversal'
    replay=state.replay_buffer_state;size=int(jax.device_get(replay.buffer_size))
    capacity=int(jax.tree.leaves(replay.data)[0].shape[0])
    rows=[];arrays={};legacy_check=hashlib.sha256();canonical=hashlib.sha256()
    for i,((path,leaf),(jax_path,jax_leaf)) in enumerate(zip(semantic,jax_leaves)):
        assert leaf is jax_leaf or np.array_equal(_array(leaf),_array(jax_leaf)),path
        a=_array(leaf);key=f'leaf_{i:05d}';arrays[key]=a
        row=dict(path=path,legacy_path=jax.tree_util.keystr(jax_path),group=_group(path),array_key=key,**_summary(a))
        if path.startswith('replay_buffer_state.') and a.ndim and a.shape[0]==capacity:
            row['resident']=_summary(a[:size]);row['unused']=_summary(a[size:])
        rows.append(row)
        legacy_check.update(row['legacy_path'].encode());legacy_check.update(str((a.dtype,a.shape)).encode());legacy_check.update(a.tobytes())
    assert legacy_check.hexdigest()==legacy,'legacy fingerprint serialization mismatch'
    for row in sorted(rows,key=lambda r:r['path']):
        canonical.update(row['path'].encode());canonical.update(str((row['dtype'],row['shape'])).encode());canonical.update(arrays[row['array_key']].tobytes())
    np.savez_compressed(directory/'leaves.npz',**arrays)
    result=dict(schema=1,legacy_sha256=legacy,canonical_sha256=canonical.hexdigest(),replay_buffer_size=size,replay_capacity=capacity,metadata=metadata,leaves=rows)
    with (directory/'evidence.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
    with (directory/'checksums.sha256').open('x') as f:
        for name in ['evidence.json','leaves.npz']:f.write(f'{hashlib.sha256((directory/name).read_bytes()).hexdigest()}  {name}\n')
    return result

def _verify(directory):
    for line in (directory/'checksums.sha256').read_text().splitlines():
        expected,name=line.split('  ',1)
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise ValueError('evidence checksum mismatch')

def compare_evidence(left,right):
    left=Path(left);right=Path(right);_verify(left);_verify(right)
    a=json.loads((left/'evidence.json').read_text());b=json.loads((right/'evidence.json').read_text())
    aa={r['path']:r for r in a['leaves']};bb={r['path']:r for r in b['leaves']};differences=[]
    with np.load(left/'leaves.npz',allow_pickle=False) as az,np.load(right/'leaves.npz',allow_pickle=False) as bz:
        for path in sorted(set(aa)|set(bb)):
            x=aa.get(path);y=bb.get(path)
            if x is None or y is None:differences.append(dict(path=path,classification='missing_leaf'));continue
            if (x['shape'],x['dtype'],x['sha256'])==(y['shape'],y['dtype'],y['sha256']):continue
            row=dict(path=path,group=x['group'],left_sha256=x['sha256'],right_sha256=y['sha256'])
            if (x['shape'],x['dtype'])!=(y['shape'],y['dtype']):row['classification']='shape_or_dtype_difference'
            else:
                av=az[x['array_key']];bv=bz[y['array_key']]
                equal=np.array_equal(av,bv,equal_nan=True)
                row['classification']='byte_representation_difference' if equal else 'numeric_value_difference'
                row['different_values']=int(np.sum(av!=bv))
                if av.size and np.issubdtype(av.dtype,np.number):
                    delta=np.abs(av.astype(np.float64)-bv.astype(np.float64));finite=delta[np.isfinite(delta)]
                    row['max_finite_abs']=float(finite.max()) if finite.size else None
            for region in ['resident','unused']:
                if region in x and region in y:row[region+'_equal']=x[region]==y[region]
            differences.append(row)
    return dict(legacy_equal=a['legacy_sha256']==b['legacy_sha256'],canonical_equal=a['canonical_sha256']==b['canonical_sha256'],different_leaves=len(differences),differences=differences)
