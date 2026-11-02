import json
from pathlib import Path
import pytest
from hydra import compose,initialize
from omegaconf import OmegaConf
from scripts.calibrate_utility_replay import registration
from scripts.collect_utility_replay_calibration import hashes
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_pipeline import PROTOCOL
from evorl.algorithms.erl.erl_td3.utility_replay_calibration_offline_v2 import digest
import scripts.analyze_utility_replay_calibration_v2 as runner


def corpus(tmp_path):
    root=tmp_path/'corpus';(root/'snapshots').mkdir(parents=True);(root/'logs').mkdir()
    for generation in range(11,71):
        (root/'snapshots'/f'generation_{generation:06d}.json').write_text(json.dumps({'generation':generation}))
        (root/'snapshots'/f'generation_{generation:06d}.npz').write_bytes(b'test-only-not-simulated')
    hashes(root/'snapshots','snapshots.sha256')
    prereg=dict(artifact_root=str(root),protocol=PROTOCOL,commit='0'*40,calibration_seed=104729,
                formal_seeds=[1,2,3],protocol_sha256=digest(PROTOCOL))
    (root/'preregistration.json').write_text(json.dumps(prereg))
    (root/'candidates.json').write_text(json.dumps(registration()))
    (root/'collection_config.yaml').write_text('test_only: true\n')
    (root/'runtime.json').write_text(json.dumps({'test_only':True}))
    (root/'collection_counters.json').write_text(json.dumps({'episodes':2300,'generation':70}))
    (root/'logs/collection.exit').write_text('0\n')
    hashes(root,'preregistration.sha256');hashes(root,'preflight.sha256')
    with initialize(config_path='../configs',version_base=None):
        config=compose(config_name='config',overrides=['agent=erl/erl-utility-replay','env=brax/swimmer','seed=1'])
    base=tmp_path/'formal.yaml';OmegaConf.save(config,base,resolve=True)
    return root,base,runner.sha(root/'snapshots/snapshots.sha256')


def setup_runtime(monkeypatch,tmp_path,commit='1'*40):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'require_clean_tree',lambda:None)
    monkeypatch.setattr(runner,'git_head',lambda:commit)
    monkeypatch.setattr(runner,'runtime',lambda:dict(backend='gpu',version='test-only'))


def test_new_analysis_records_both_commits_and_preserves_source(tmp_path,monkeypatch):
    root,base,sha=corpus(tmp_path);setup_runtime(monkeypatch,tmp_path)
    before=(root/'preregistration.json').read_bytes()
    result=runner.prepare(root,base,tmp_path/'work-v2',tmp_path/'selected-v2',sha)
    assert result['collection_commit']=='0'*40 and result['analysis_commit']=='1'*40
    assert result['policy']==registration() and result['analysis_generations']==list(range(20,71))
    assert result['draws']==1024 and result['code_sha256']
    assert result['collection_files_sha256']['collection_config.yaml']
    assert runner.load_registration(tmp_path/'work-v2')['corpus_manifest_sha256']==sha
    assert (root/'preregistration.json').read_bytes()==before
    assert not (tmp_path/'selected-v2').exists()
    with pytest.raises(FileExistsError): runner.prepare(root,base,tmp_path/'work-v2',tmp_path/'selected-v2',sha)


def test_corpus_checksum_or_same_implementation_commit_is_rejected(tmp_path,monkeypatch):
    root,base,sha=corpus(tmp_path);setup_runtime(monkeypatch,tmp_path)
    with pytest.raises(ValueError): runner.prepare(root,base,tmp_path/'bad-sha',tmp_path/'out','f'*64)
    setup_runtime(monkeypatch,tmp_path,commit='0'*40)
    with pytest.raises(ValueError,match='distinct'): runner.prepare(root,base,tmp_path/'bad-version',tmp_path/'out',sha)


def test_version_registration_tampering_is_rejected(tmp_path,monkeypatch):
    root,base,sha=corpus(tmp_path);setup_runtime(monkeypatch,tmp_path)
    runner.prepare(root,base,tmp_path/'work',tmp_path/'out',sha)
    path=tmp_path/'work/analysis_registration.json';data=json.loads(path.read_text())
    data['analysis_commit']='2'*40;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='corrupted'): runner.load_registration(tmp_path/'work')
