from .individual import Individual
from .stage1_individual import Stage1Individual
from .stage2_individual import Stage2Individual
from .stage3_individual import Stage3Individual
from .stage3_single_episode import Stage3SingleEpisodeIndividual
from .collector_individual import CollectorIndividual

REGISTRY = {
    'stage1': Stage1Individual,
    'stage2': Stage2Individual,
    'stage3': Stage3Individual,
    'collector': CollectorIndividual,
    'stage3_se': Stage3SingleEpisodeIndividual,
}
