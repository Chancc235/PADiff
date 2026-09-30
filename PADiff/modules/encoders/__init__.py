from .gru_encoder import GRUEncoder
from .gru_vi_encoder import GRUVIEncoder
from .lstm_encoder import LSTMEncoder

REGISTRY = {
    "rnn": GRUEncoder,
    "rnn_vi": GRUVIEncoder,
    "lstm": LSTMEncoder,
}
