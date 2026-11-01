from types import SimpleNamespace

import pytest
import jax.numpy as jnp
import optax
from hydra import compose, initialize

from evorl.algorithms.erl.erl_td3.erl_utility_replay import ERLUtilityReplayWorkflow, _optimizer_step_count

from evorl.algorithms.erl.erl_td3.utility_replay_budget import (
    BudgetAudit,
    prepare_arm_manifest,
    validate_arm_manifest,
    validate_arm_set,
)


def _arm(arm):
    return SimpleNamespace(
        arm=arm,
        score_kind={"uniform": "uniform", "fitness": "fitness", "td": "td", "utility": "utility"}[arm],
        alpha=0.0 if arm == "uniform" else 0.5,
        replay_temperature=1.0,
        freshness_window=3,
        min_resident=4,
        diagnostic_enabled=arm != "uniform",
        diagnostic_interval=1,
        total_episodes=100,
        episodes_for_fitness=1,
        pop_size=4,
        rollout_episodes=1,
        num_rl_agents=1,
        batch_size=8,
        actor_update_interval=2,
        rl_updates_frac=1.0,
        rl_updates_mode="iter",
        eval_interval=2,
        eval_episodes=3,
        num_eval_envs=3,
        num_elites=1,
        tournament_size=3,
        rl_injection_interval=10,
        replay_buffer_capacity=64,
    )


def _arms():
    return [_arm(x) for x in ("uniform", "fitness", "td", "utility")]


def _completed_audits():
    configs = _arms()
    validate_arm_set(configs)
    audits = []
    for config in configs:
        audit = BudgetAudit(config, initial_episodes=0, initial_iteration=0)
        for iteration in range(1, 21):
            audit.observe_step(episodes=5, timesteps=20, rl_updates=10, batch_size=8, ea_ask=1, ea_tell=1, actor_optimizer_steps=10, critic_optimizer_steps=20)
            if iteration % 2 == 0:
                audit.observe_eval(iteration, episodes=3)
        audits.append(audit.finalize(final_episodes=100, final_iteration=20, final_timesteps=400))
    return audits


def test_abcd_arms_have_equal_environment_interactions():
    assert len({a["environment_episodes"] for a in _completed_audits()}) == 1
    assert len({a["environment_timesteps"] for a in _completed_audits()}) == 1


def test_abcd_arms_have_equal_learner_update_counts():
    assert {a["rl_updates"] for a in _completed_audits()} == {200}


def test_abcd_arms_have_equal_batch_size():
    assert {a["batch_size"] for a in _completed_audits()} == {8}


def test_abcd_arms_have_equal_population_operations():
    assert {(a["ea_ask"], a["ea_tell"]) for a in _completed_audits()} == {(20, 20)}


def test_abcd_arms_have_equal_evaluation_checkpoints():
    assert {tuple(a["evaluation_checkpoints"]) for a in _completed_audits()} == {tuple(range(2, 21, 2))}


def test_abcd_arms_have_equal_evaluation_episode_counts():
    assert {a["evaluation_episodes"] for a in _completed_audits()} == {30}


def test_abcd_arms_have_equal_total_training_budget():
    assert {a["configured_total_episodes"] for a in _completed_audits()} == {100}


def test_all_arms_emit_common_audit_schema():
    audits = _completed_audits()
    assert all(set(a) == set(audits[0]) for a in audits)


@pytest.mark.parametrize("field,value", [
    ("alpha", 0.7), ("replay_temperature", 2.0),
    ("freshness_window", 4), ("min_resident", 5),
    ("total_episodes", 101), ("batch_size", 9),
])
def test_arm_specific_budget_or_sampler_override_is_rejected(field, value):
    configs = _arms()
    setattr(configs[2], field, value)
    with pytest.raises(ValueError):
        validate_arm_set(configs)


def test_wrong_arm_score_kind_is_rejected():
    configs = _arms()
    configs[3].score_kind = "fitness"
    with pytest.raises(ValueError):
        validate_arm_set(configs)


def test_actual_counter_mismatch_is_rejected():
    config = _arm("fitness")
    audit = BudgetAudit(config, initial_episodes=0, initial_iteration=0)
    audit.observe_step(episodes=5, timesteps=20, rl_updates=7, batch_size=8, ea_ask=1, ea_tell=1)
    with pytest.raises(ValueError):
        audit.finalize(final_episodes=5, final_iteration=1, final_timesteps=20)


def test_frozen_manifest_accepts_each_resolved_arm(tmp_path):
    configs = _arms()
    path = tmp_path / "arms.json"
    prepare_arm_manifest(configs, path)
    for config in configs:
        config.arm_manifest_path = str(path)
        validate_arm_manifest(config)


def test_frozen_manifest_rejects_independent_arm_budget_change(tmp_path):
    configs = _arms()
    path = tmp_path / "arms.json"
    prepare_arm_manifest(configs, path)
    configs[3].arm_manifest_path = str(path)
    configs[3].eval_interval = 3
    with pytest.raises(ValueError):
        validate_arm_manifest(configs[3])


def test_formal_four_arm_configs_build_and_share_frozen_manifest(tmp_path):
    configs = []
    with initialize(config_path="../configs", version_base=None):
        for arm in ("uniform", "fitness", "td", "utility"):
            configs.append(compose(config_name="config", overrides=[
                "env=brax/swimmer", "agent=erl/erl-utility-replay",
                f"arm={arm}", f"score_kind={arm}",
                f"alpha={'0' if arm == 'uniform' else '0.5'}",
                f"diagnostic_enabled={'false' if arm == 'uniform' else 'true'}",
            ]))
    path = tmp_path / "frozen_arms.json"
    prepare_arm_manifest(configs, path)
    for arm, config in zip(("uniform", "fitness", "td", "utility"), configs):
        config.arm_manifest_path = str(path)
        validate_arm_manifest(config)
        workflow = ERLUtilityReplayWorkflow.build_from_config(config, enable_jit=False)
        assert workflow.config.arm == arm
        assert workflow.config.score_kind == arm
        assert float(workflow.config.alpha) == (0.0 if arm == "uniform" else 0.5)


def test_optimizer_counter_reads_real_optax_state():
    optimizer = optax.adam(0.001)
    params = jnp.asarray([1.0])
    state = optimizer.init(params)
    assert _optimizer_step_count(state) == 0
    _, state = optimizer.update(jnp.asarray([0.1]), state, params)
    assert _optimizer_step_count(state) == 1


def test_eval_episode_audit_uses_actual_rounded_result_count():
    config = _arm("fitness")
    config.eval_episodes = 3
    config.num_eval_envs = 2
    audit = BudgetAudit(config, initial_episodes=0, initial_iteration=0)
    audit.observe_eval(2, episodes=4)
    with pytest.raises(ValueError):
        audit.observe_eval(4, episodes=3)


def test_alpha_zero_jitted_evaluation_matches_original_erl():
    import chex
    import jax
    from evorl.algorithms.erl.erl_td3.erl_origin import ERLWorkflow

    common = [
        "env=brax/swimmer", "env.max_episode_steps=1", "num_envs=1",
        "pop_size=4", "num_elites=1", "num_rl_agents=1",
        "episodes_for_fitness=1", "rollout_episodes=1", "batch_size=4",
        "replay_buffer_capacity=64", "warmup_iters=0", "num_eval_envs=2", "eval_episodes=3",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
    ]
    with initialize(config_path="../configs", version_base=None):
        original_cfg = compose(config_name="config", overrides=["agent=erl/erl-ori", *common])
        pilot_cfg = compose(config_name="config", overrides=["agent=erl/erl-utility-replay", *common])
    original = ERLWorkflow.build_from_config(original_cfg, enable_jit=True)
    pilot = ERLUtilityReplayWorkflow.build_from_config(pilot_cfg, enable_jit=True)
    key = jax.random.PRNGKey(41)
    original_state = original.setup(key)
    pilot_state = pilot.setup(key)
    original_metrics, original_state = original.evaluate(original_state)
    pilot_metrics, pilot_state = pilot.evaluate(pilot_state)
    chex.assert_trees_all_equal(original_metrics, pilot_metrics)
    chex.assert_trees_all_equal(original_state.key, pilot_state.key)


def test_ten_jitted_warmup_steps_preserve_original_state_and_provenance():
    import chex
    import jax
    import jax.numpy as jnp
    from evorl.algorithms.erl.erl_td3.erl_origin import ERLWorkflow

    common = [
        "env=brax/swimmer", "env.max_episode_steps=1", "num_envs=1",
        "pop_size=4", "num_elites=1", "num_rl_agents=1",
        "episodes_for_fitness=1", "rollout_episodes=1", "batch_size=4",
        "replay_buffer_capacity=64", "warmup_iters=10", "eval_interval=1",
        "num_eval_envs=1", "eval_episodes=1",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
    ]
    with initialize(config_path="../configs", version_base=None):
        original_cfg = compose(config_name="config", overrides=["agent=erl/erl-ori", *common])
        pilot_cfg = compose(config_name="config", overrides=["agent=erl/erl-utility-replay", *common])
    original = ERLWorkflow.build_from_config(original_cfg, enable_jit=True)
    pilot = ERLUtilityReplayWorkflow.build_from_config(pilot_cfg, enable_jit=True)
    key = jax.random.PRNGKey(515)
    base_state = original.setup(key)
    pilot_state = pilot.setup(key)
    for field in ("key", "agent_state", "opt_state", "ec_opt_state", "metrics"):
        chex.assert_trees_all_equal(getattr(base_state, field), getattr(pilot_state, field))
    for field in ("data", "current_index", "buffer_size"):
        chex.assert_trees_all_equal(getattr(base_state.replay_buffer_state, field), getattr(pilot_state.replay_buffer_state, field))
    assert int(pilot_state.metrics.iterations) == 10
    replay = pilot_state.replay_buffer_state
    assert set(map(int, jnp.unique(replay.source_generation[:40]))) == set(range(1, 11))
    assert set(map(int, jnp.unique(replay.source_token[:40]))) == set(range(4, 44))


def test_all_four_frozen_arms_complete_ten_jitted_warmup_steps(tmp_path):
    import jax

    overrides = [
        "env=brax/swimmer", "env.max_episode_steps=1", "num_envs=1",
        "pop_size=4", "num_elites=1", "num_rl_agents=1",
        "episodes_for_fitness=1", "rollout_episodes=1", "batch_size=4",
        "replay_buffer_capacity=64", "warmup_iters=10", "eval_interval=1",
        "num_eval_envs=1", "eval_episodes=1",
        "agent_network.critic_hidden_layer_sizes=[8,8]",
        "agent_network.actor_hidden_layer_sizes=[8,8]",
    ]
    configs = []
    with initialize(config_path="../configs", version_base=None):
        for arm in ("uniform", "fitness", "td", "utility"):
            configs.append(compose(config_name="config", overrides=[
                "agent=erl/erl-utility-replay", f"arm={arm}", f"score_kind={arm}",
                f"alpha={'0' if arm == 'uniform' else '0.5'}",
                f"diagnostic_enabled={'false' if arm == 'uniform' else 'true'}",
                *overrides,
            ]))
    path = tmp_path / "arms.json"
    prepare_arm_manifest(configs, path)
    for arm, config in zip(("uniform", "fitness", "td", "utility"), configs):
        config.arm_manifest_path = str(path)
        validate_arm_manifest(config)
        workflow_cls = type(f"Pilot_{arm}", (ERLUtilityReplayWorkflow,), {})
        workflow = workflow_cls.build_from_config(config, enable_jit=True)
        state = workflow.setup(jax.random.PRNGKey(611))
        assert int(state.metrics.iterations) == 10
        assert int(state.replay_buffer_state.buffer_size) == 40


def test_warmup_source_token_int32_overflow_is_rejected_before_jit():
    with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="config", overrides=[
            "env=brax/swimmer", "agent=erl/erl-utility-replay",
            "warmup_iters=2147483647", "pop_size=4",
        ])
    with pytest.raises(ValueError, match="source_token"):
        ERLUtilityReplayWorkflow.build_from_config(config, enable_jit=False)


@pytest.mark.parametrize("arm", ("uniform", "utility"))
def test_runtime_budget_audit_observes_real_one_step_workflow(monkeypatch, arm):
    import jax
    from evorl.algorithms.erl.erl_td3.erl_ga import ERLGAWorkflow

    with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="config", overrides=[
            "env=brax/swimmer", "env.max_episode_steps=1",
            "agent=erl/erl-utility-replay", f"arm={arm}", f"score_kind={arm}",
            f"alpha={'0' if arm == 'uniform' else '0.5'}",
            f"diagnostic_enabled={'false' if arm == 'uniform' else 'true'}",
            "num_envs=1", "pop_size=4", "num_elites=1", "num_rl_agents=1",
            "episodes_for_fitness=1", "rollout_episodes=1", "total_episodes=5",
            "batch_size=4", "replay_buffer_capacity=64", "warmup_iters=0",
            "actor_update_interval=1", "rl_updates_frac=0.25",
            "agent_network.critic_hidden_layer_sizes=[8,8]",
            "agent_network.actor_hidden_layer_sizes=[8,8]",
            "num_eval_envs=2", "eval_episodes=3", "eval_interval=1",
        ])
    workflow = ERLUtilityReplayWorkflow.build_from_config(config, enable_jit=False)

    def one_step_with_evaluation(self, state):
        _, state = self.step(state)
        _, state = self.evaluate(state)
        return state

    monkeypatch.setattr(ERLGAWorkflow, "learn", one_step_with_evaluation)
    state = workflow.setup(jax.random.PRNGKey(101))
    workflow.learn(state)
    audit = workflow._last_budget_audit
    assert audit["environment_episodes"] == 5
    assert audit["ea_ask"] == audit["ea_tell"] == 1
    assert audit["evaluation_episodes"] == 4
    assert audit["evaluation_checkpoints"] == (1,)


@pytest.mark.parametrize("kind", ("episodes", "timesteps", "updates", "ask", "tell", "eval"))
def test_actual_observation_mismatch_is_rejected(kind):
    config = _arm("fitness")
    config.total_episodes = 5
    audit = BudgetAudit(config, initial_episodes=0, initial_iteration=0)
    kwargs = dict(episodes=5, timesteps=20, rl_updates=10, batch_size=8, ea_ask=1, ea_tell=1, actor_optimizer_steps=10, critic_optimizer_steps=20)
    if kind in ("episodes", "updates", "ask", "tell"):
        kwargs[{"episodes": "episodes", "updates": "rl_updates", "ask": "ea_ask", "tell": "ea_tell"}[kind]] -= 1
        with pytest.raises(ValueError):
            audit.observe_step(**kwargs)
        return
    audit.observe_step(**kwargs)
    if kind == "eval":
        with pytest.raises(ValueError):
            audit.observe_eval(1, episodes=2)
    else:
        audit.observe_eval(1, episodes=3)
        with pytest.raises(ValueError):
            audit.finalize(final_episodes=5, final_iteration=1, final_timesteps=21)
