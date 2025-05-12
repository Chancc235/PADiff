from .q_learner import QLearner
from .q_plus_learner import QPlusLearner
from .stage2_learner import Stage2Learner
from .ODITS_learner import ODITSLearner

REGISTRY = {
    "q_learner": QLearner,
    "q+": QPlusLearner,
    'stage2': Stage2Learner,
    "ODITS": ODITSLearner,
}
