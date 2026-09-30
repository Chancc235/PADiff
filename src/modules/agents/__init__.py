from .context_rnn_agent import ContextRNNAgent
from .FIAM_agent import FIAMAgent
from .kmeans_multimodule_agent import KmeansRnnAgent
from .LIAM_agent import LIAMAgent
from .ODITS_agent import ODITSAgent
from .rnn_agent import RNNAgent
from .rule_agent_lbf import LBFRuleAgent
from .rule_agent_overcooked import OvercookedRuleAgent

REGISTRY = {
    "rnn": RNNAgent,
    "context_rnn": ContextRNNAgent,
    "overcooked_rule": OvercookedRuleAgent,
    "lbf_rule": LBFRuleAgent,
    "LIAM": LIAMAgent,
    "FIAM": FIAMAgent,
    "ODITS": ODITSAgent,
    "kmeans": KmeansRnnAgent,
}
