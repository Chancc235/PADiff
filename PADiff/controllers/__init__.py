from .basic_controller import BasicMAC
from .context_controller import ContextMAC_v1, ContextMAC_v2, ContextMAC_v3, ContextMAC_v4
from .partial_controller import PartialMAC, PartialRuleMAC

REGISTRY = {
    "basic": BasicMAC,
    "partial": PartialMAC,
    "partial_rule": PartialRuleMAC,
    "cmac_v1": ContextMAC_v1,
    "cmac_v2": ContextMAC_v2,
    "cmac_v3": ContextMAC_v3,
    "cmac_v4": ContextMAC_v4,
}
