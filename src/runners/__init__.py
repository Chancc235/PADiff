from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner
from .meta_runner import MetaRunner, ParallelProcessor

REGISTRY = {
    "episode": EpisodeRunner,
    "parallel": ParallelRunner,
    "meta": MetaRunner,
}
