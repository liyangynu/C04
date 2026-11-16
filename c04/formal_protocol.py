"""Shared scope checks for the fixed 10-shot, 30% comparisons."""

SEEDS = [17, 42, 2026]
METHODS = ['clean_ce', 'ce', 'gce', 'cl_ce']


def validate_source(source):
    if (source['noise'] not in {'instance', 'symmetric'} or source['rate'] != .3
            or source['shots_per_class'] != [10] or source['seeds'] != SEEDS):
        raise ValueError('Expected a fixed three-seed 30% noise, 10-shot cohort')
    if sorted(r['seed'] for r in source['records']) != sorted(SEEDS):
        raise ValueError('Source must contain exactly one record per fixed seed')


def comparison_noise(comparison):
    # The first published v2 comparison predates explicit noise metadata.
    noise, rate = comparison.get('noise', 'instance'), comparison.get('rate', .3)
    if noise not in {'instance', 'symmetric'} or rate != .3:
        raise ValueError('Unsupported formal comparison noise protocol')
    records = comparison['records']
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    keys = [(r['method'], r['seed']) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Comparison must have all 12 unique method/seed pairs')
    if any(r['shots_per_class'] != 10 for r in records):
        raise ValueError('Expected 10-shot comparison')
    return noise, rate
