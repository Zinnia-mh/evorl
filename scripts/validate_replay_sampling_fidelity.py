"""CPU-only exact sampling-law audit of sealed D0 historical checkpoints.

No models, TD3 updates or environment rollouts are executed. The binary search
partition counts uniform grid points exactly, including nonmonotone old CDFs.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
from pathlib import Path
import json,hashlib,re,time
import jax,numpy as np,orbax.checkpoint as ocp
ROOT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/replication/swimmer/seed_2047/ad300-head43221c7-v1")
OUT=Path("/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/analysis/d0-sampler-fidelity-v1")
def path_value(raw,path):
 for flat,quoted,index,attr in re.findall(r"\[<flat index (\d+)>\]|\['([^']+)'\]|\[(\d+)\]|\.([A-Za-z_][A-Za-z_0-9]*)",path):
  raw=raw[int(flat or index)] if flat or index else raw[quoted or attr]
 return raw
def exact_binary_counts(cdf,bits):
 """Exact grid cardinalities for JAX scan searchsorted, including nonmonotone CDF."""
 scale=2**bits;n=len(cdf)
 lo=np.array([0],np.int64);hi=np.array([n],np.int64)
 start=np.array([0],np.int64);end=np.array([scale],np.int64)
 cuts=np.clip(np.ceil(np.ldexp(cdf.astype(np.float64),bits)),0,scale).astype(np.int64)
 for _ in range(int(np.ceil(np.log2(n+1)))):
  mid=(lo+hi)//2;cut=cuts[mid]
  left_end=np.minimum(end,cut);right_start=np.maximum(start,cut)
  left=left_end>start;right=end>right_start
  lo,hi,start,end=(np.r_[lo[left],mid[right]],np.r_[mid[left],hi[right]],
   np.r_[start[left],right_start[right]],np.r_[left_end[left],end[right]])
 counts=np.zeros(n,np.int64);np.add.at(counts,np.minimum(hi,n-1),end-start)
 assert int(counts.sum())==scale
 return counts
def source_law(cdf,bits,tokens,ea,source_ids,baseline):
 counts=exact_binary_counts(cdf,bits);_,inverse=np.unique(tokens[ea],return_inverse=True)
 mass=np.bincount(inverse,weights=counts[ea]/2**bits)
 invalid=float(counts[~ea].sum()/2**bits)
 return mass+invalid*baseline,invalid
def restored(g):
 cp=next((ROOT/"runs/D0/segments").glob(f"*/checkpoints/{g}"))
 metadata=json.loads((cp/"runtime/metadata").read_text())
 with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as c:
  raw=c.restore(str(cp/"state"),args=ocp.args.StandardRestore(fallback_sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])))
 digest=hashlib.sha256()
 for s in metadata["leaf_specs"]:
  v=np.asarray(path_value(raw,s["path"]));assert list(v.shape)==s["shape"] and str(v.dtype)==s["dtype"]
  digest.update(s["path"].encode());digest.update(str((v.dtype,v.shape)).encode());digest.update(v.tobytes())
 assert digest.hexdigest()==metadata["state_sha256"]
 return raw
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--input-root",type=Path,default=ROOT)
 parser.add_argument("--output",type=Path,required=True)
 args=parser.parse_args()
 global OUT
 input_root=args.input_root.resolve()
 OUT=args.output.resolve()
 if OUT.is_relative_to(input_root):
  raise ValueError("output must be outside frozen input artifacts")
 OUT.mkdir(parents=True,exist_ok=False)
 globals()["ROOT"]=input_root
 if jax.default_backend()!="cpu" or jax.config.jax_enable_x64:
  raise ValueError("validation requires CPU and globally disabled x64")
 import jax.numpy as jnp
 from evorl.algorithms.erl.erl_td3.source_replay import build_sampling_cdf
 results={}
 for g in (50,300):
  raw=restored(g);d=raw[6]["distribution"];t=np.asarray(d["transition_probability"])
  tokens=np.asarray(raw[5]["source_token"]);ea=np.asarray(raw[5]["origin_type"])==0
  ids,inv=np.unique(tokens[ea],return_inverse=True)
  baseline=np.bincount(inv).astype(np.float64);baseline/=baseline.sum()
  table=raw[6]["score_table"]
  lookup={int(k):i for i,k in enumerate(np.asarray(table["source_token"])) if k>=0}
  target=np.array([np.asarray(d["source_probability"])[lookup[int(k)]] for k in ids],dtype=np.float64)
  target/=target.sum()
  eligible=np.array([np.asarray(table["weighted_eligible"])[lookup[int(k)]] for k in ids])
  old=np.asarray(d["transition_cdf"])
  new=np.asarray(jax.jit(build_sampling_cdf)(jnp.asarray(t)))
  assert new.dtype==np.float64 and not jax.config.jax_enable_x64
  intervals=np.diff(np.r_[0.,new])
  assert np.all(intervals>=0) and np.all(intervals[~ea]==0) and new[-1]==1.
  rows=[]
  for name,cdf,bits in (("old",old,23),("cdf_only",new,23),
                        ("rng_only",old,52),("fixed",new,52)):
   law,invalid=source_law(cdf,bits,tokens,ea,ids,baseline)
   row=dict(case=name,uniform_bits=bits,source_l1=float(np.abs(law-target).sum()),
    tv_error=float(np.abs(law-target).sum()/2),ea_conditioned_tv_uniform=float(np.abs(law-baseline).sum()/2),
    eligible_mass=float(law[eligible].sum()),target_eligible_mass=float(target[eligible].sum()),
    noneligible_source_l1=float(np.abs(law[~eligible]-target[~eligible]).sum()),
    invalid_candidate_rate=invalid,final_invalid_rate=0.0)
   rows.append(row)
   if name=="fixed":
    assert row["source_l1"]<1e-7 and row["source_l1"]<rows[0]["source_l1"]*1e-4
    assert invalid==0
  np.savez_compressed(OUT/f"cdf-{g}.npz",old=old,fixed=new,target=target,baseline=baseline,source_ids=ids,eligible=eligible)
  results[g]=dict(checkpoint=str(next((ROOT/"runs/D0/segments").glob(f"*/checkpoints/{g}"))),
   full_state_sha_verified=True,raw_target_mass=float(np.asarray(d["source_probability"],dtype=np.float64).sum()),
   uniform=dict(tv_uniform=0.0,eligible_mass=float(baseline[eligible].sum())),
   cdf_extra_bytes=int(new.nbytes-old.nbytes),cases=rows)
  print(g,json.dumps(results[g]),flush=True)
 (OUT/"validation.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()
