"""Host-boundary full recount, protecting every still-resident identity first.

Only refresh uses this path. Incremental overwrite primitives remain unchanged.
Grouping is O(C log C); registry lookup/allocation is O(S + U), not O(C*S).
No learner keys, payload, scores or shapes are changed.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .source_replay import ResidentCountUpdateResult,SCORE_TABLE_OK,SCORE_TABLE_ERROR_CAPACITY


def recount_resident_sources(table,tokens,mask):
    source=np.array(jax.device_get(table.source_token),copy=True)
    tokens=np.asarray(jax.device_get(tokens),dtype=np.int32)
    mask=np.asarray(jax.device_get(mask),dtype=bool)
    resident=np.where(mask,tokens,-1).astype(np.int32)
    active=resident[resident>=0]
    unique,first,multiplicity=np.unique(active,return_index=True,return_counts=True)
    order=np.argsort(first)
    lookup={}
    for slot in np.flatnonzero(source>=0):lookup.setdefault(int(source[slot]),int(slot))
    counts=np.zeros(len(source),np.int32);cleared=np.zeros(len(source),bool);cursor=0
    # Physical ring order can put newly written tokens before old, still-live
    # identities. Establish their final counts before reclaiming ANY slot.
    # A zero running count is not evidence that its source has left Replay.
    for token,count in zip(unique,multiplicity):
        slot=lookup.get(int(token))
        if slot is not None:counts[slot]=int(count)
    for i in order:
        token=int(unique[i]);slot=lookup.get(token)
        if slot is None:
            while cursor<len(source) and counts[cursor]>0:cursor+=1
            if cursor==len(source):
                reset=table._replace(resident_count=jnp.zeros_like(table.resident_count))
                return ResidentCountUpdateResult(reset,jnp.full(resident.shape,-1,jnp.int32),jnp.int8(SCORE_TABLE_ERROR_CAPACITY))
            slot=cursor;previous=int(source[slot])
            if lookup.get(previous)==slot:del lookup[previous]
            source[slot]=token;lookup[token]=slot;cleared[slot]=True
            counts[slot]=int(multiplicity[i])
    clear=jnp.asarray(cleared)
    replacements=dict(source_token=jnp.asarray(source),resident_count=jnp.asarray(counts))
    for field,empty in [('raw_score',0.),('rank_normalized_score',0.),('measured_generation',-1),('score_kind',0),('validity',False),('weighted_eligible',False),('version',-1)]:
        replacements[field]=jnp.where(clear,jnp.asarray(empty,dtype=getattr(table,field).dtype),getattr(table,field))
    return ResidentCountUpdateResult(table._replace(**replacements),jnp.asarray(resident),jnp.int8(SCORE_TABLE_OK))
