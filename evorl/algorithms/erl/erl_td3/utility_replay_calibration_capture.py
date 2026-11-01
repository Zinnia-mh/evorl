"""Read-only calibration snapshot boundary; contains no rollout or learner calls."""
import hashlib
import json
from pathlib import Path
import numpy as np


def seed_preflight(formal_seeds, calibration_seed=104729):
    if len(formal_seeds) != 3 or len(set(formal_seeds)) != 3:
        raise ValueError('three explicit distinct formal seeds must be registered')
    if calibration_seed in formal_seeds:
        raise ValueError('calibration seed conflicts with formal seeds')
    return dict(calibration_seed=calibration_seed, formal_seeds=list(formal_seeds))


def capture(replay, records, generation, *, population_size, timing, memory_mib):
    """Copy real sidecars; caller retains original replay and training key.

    Invoke once per generation, after all insertions at a fixed boundary.
    Records are the frozen pre-exposure diagnostic output, never returns.
    No missing runtime measurement is replaced with an estimate.
    """
    if set(timing) != {'diagnostic_ms','sampling_ms'} or any(v < 0 or not np.isfinite(v) for v in timing.values()):
        raise ValueError('independent measured diagnostic/sampling times required')
    if memory_mib < 0 or not np.isfinite(memory_mib):
        raise ValueError('measured memory required')
    arrays = {name:np.array(getattr(replay,name),copy=True) for name in
              ['origin_type','source_token','source_generation','score_version']}
    capacity=len(arrays['source_token']); size=int(replay.buffer_size)
    if not 0 <= size <= capacity or any(len(x)!=capacity for x in arrays.values()):
        raise ValueError('invalid replay metadata alignment')
    resident=np.arange(capacity)<size
    tokens,counts=np.unique(arrays['source_token'][resident & (arrays['origin_type']==0)],return_counts=True)
    if np.any(tokens<0): raise ValueError('resident EA source identity missing')
    ea=resident & (arrays['origin_type']==0)
    if np.any(arrays['source_generation'][ea] != arrays['source_token'][ea]//population_size):
        raise ValueError('source generation and token mismatch')
    scores=[]
    for record in records:
        measured=int(record.generation)
        if measured>generation: raise ValueError('future score leakage')
        scores.append(dict(source_token=measured*population_size+int(record.individual_id),
                           measured_generation=measured, fitness=float(record.fitness),
                           td=None if record.td_error is None else float(record.td_error),
                           utility=None if record.utility is None else float(record.utility),valid=bool(record.eligible)))
    metadata=dict(schema=1,generation=int(generation),capacity=capacity,buffer_size=size,population_size=int(population_size),
                  current_index=int(replay.current_index),resident_counts=dict(zip(map(str,tokens),map(int,counts))),
                  scores=scores,timing=dict(timing),memory_mib=float(memory_mib),
                  simulation_prng=dict(library='jax',root_seed=104729,domain_tag=0x53494D55,fold_in='generation*3+kind_id'))
    return metadata,arrays


def write_snapshot(directory, metadata, arrays):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    stem=f"generation_{metadata['generation']:06d}"
    for suffix in ['.json','.npz']:
        if (directory/(stem+suffix)).exists(): raise FileExistsError(stem)
    np.savez_compressed(directory/(stem+'.npz'),**arrays)
    (directory/(stem+'.json')).write_text(json.dumps(metadata,sort_keys=True,allow_nan=False))
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [directory/(stem+'.json'),directory/(stem+'.npz')]}
