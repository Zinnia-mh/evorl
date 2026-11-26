"""Only new A100 provenance and prospective AUC endpoints."""
import importlib,json
from pathlib import Path
import pytest

def module():
    try:return importlib.import_module("scripts.run_rl_guided_late_pilot")
    except ModuleNotFoundError as e:
        if e.name=="scripts.run_rl_guided_late_pilot":pytest.fail("M-Late runner missing")
        raise

def test_parent_is_original_uniform_A_100_not_M_100(tmp_path):
    m=module();base=tmp_path/"baseline"
    cp=base/"runs/A/segments/original/checkpoints/100"
    (cp/"runtime").mkdir(parents=True)
    (cp/"runtime/metadata").write_text(json.dumps({"generation":100,"state_sha256":"full-A-state"}))
    summary={"performance":{"segment_directory":str(cp.parent.parent)}}
    assert m.original_A_parent(summary,base)==cp
    wrong={"performance":{"segment_directory":str(tmp_path/"old-M/runs/M/segments/other")}}
    with pytest.raises(ValueError,match="original A"):m.original_A_parent(wrong,base)
    (cp/"runtime/metadata").write_text(json.dumps({"generation":50,"state_sha256":"wrong-generation"}))
    with pytest.raises(ValueError,match="CP100"):m.original_A_parent(summary,base)

def test_primary_auc_includes_shared100_and_excludes_earlier_prefix():
    m=module()
    rows=[dict(generation=90,environment_timesteps=0,return_mean=9999),
          dict(generation=100,environment_timesteps=3290000,return_mean=100),
          dict(generation=110,environment_timesteps=3620000,return_mean=200),
          dict(generation=150,environment_timesteps=4940000,return_mean=300)]
    assert m.interval_auc(rows)==330000*150+1320000*250
