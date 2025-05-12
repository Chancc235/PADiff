from .population import Population, StrPopulation
from .stage1_population import Stage1Population
from .stage2_population import Stage2Population
from .stage3_population import Stage3Population
from .collector_population import CollectorPopulation

# from .test_population import TestPopulation

REGISTRY = {
    'stage1': Stage1Population,
    'stage2': Stage2Population,
    'stage3': Stage3Population,
    'collector': CollectorPopulation,
    # 'test': TestPopulation,
}
