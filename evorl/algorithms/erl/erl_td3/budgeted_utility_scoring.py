"""Experiment-only equal-budget distributed pre-exposure Utility scoring."""
import jax
import numpy as np
from .learner_utility import derive_diagnostic_key
from .erl_utility_replay import ERLUtilityReplayWorkflow

SCHEDULE_TAG = 0x42534348

def score_quota(generation, population_size, interval, warmup_iters):
    generation, population_size, interval, warmup_iters = map(
        int, (generation, population_size, interval, warmup_iters))
    if population_size < 1 or interval < 1 or warmup_iters < 0:
        raise ValueError("invalid distributed scoring budget")
    step = max(0, generation - warmup_iters)
    if not step:
        return 0
    return (step * population_size // interval
            - (step - 1) * population_size // interval)

class DistributedScoringMixin:
    def _diagnostic_due(self, state):
        if not getattr(self, "_diagnostic_enabled", False):
            return False
        return score_quota(int(jax.device_get(state.metrics.iterations)) + 1,
            self.config.pop_size, self.config.diagnostic_interval,
            self.config.warmup_iters) > 0

    def _select_diagnostic_sources(self, sources):
        context = self._diagnostic_context
        quota = score_quota(context.generation, self.config.pop_size,
            self.config.diagnostic_interval, self.config.warmup_iters)
        key = derive_diagnostic_key(context.key, context.generation, SCHEDULE_TAG)
        order = np.asarray(jax.random.permutation(key, len(sources)))[:quota]
        # Keep original identities and original utility loss/random keys.
        return tuple(sources[int(i)] for i in sorted(order))

class BudgetedUtilityReplayWorkflow(DistributedScoringMixin, ERLUtilityReplayWorkflow):
    """Same training/sampler; only the diagnostic timing and source subset vary."""
    pass
