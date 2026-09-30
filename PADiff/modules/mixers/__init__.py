from .qmix import QMixer
from .vdn import VDNMixer

REGISTRY = {
    'vdn': VDNMixer,
    "qmix": QMixer,
}
