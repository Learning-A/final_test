"""Pure feature selection and fixed objectives; fitting is server-only."""
import math

ARMS = (
    ('total5_mse', 'total5', 'squared_error', False, False),
    ('self50_mse', 'self50', 'squared_error', False, False),
    ('spatial_mse', 'spatial', 'squared_error', False, False),
    ('spatial_log_mse', 'spatial', 'squared_error', True, False),
    ('spatial_mae', 'spatial', 'absolute_error', False, False),
    ('spatial_relative_mae', 'spatial', 'absolute_error', False, True),
)


def select_features(vector, target, group):
    if not 0 <= target <= 8 or len(vector) != (432 if target == 8 else 82):
        raise ValueError('unexpected feature schema or target')
    if group == 'spatial':
        return list(vector)
    if group == 'self50':
        return list(vector[:400 if target == 8 else 50])
    if group == 'total5':
        offsets = range(0, 400, 50) if target == 8 else (0,)
        return [vector[o+i] for o in offsets for i in (2, 5, 8, 11, 14)]
    raise ValueError('unknown input group')


def relative_weights(training_labels):
    """Fixed 1 IOPS training floor, includes zeros; no development fitting.

    Weighted absolute-error trees are a relative-error candidate. The floor,
    restricted trees and ensemble do not guarantee minimum formal MAPE.
    Formal scoring never uses this denominator floor.
    """
    if not training_labels or any(not math.isfinite(y) or y < 0 for y in training_labels):
        raise ValueError('invalid training labels')
    return [1/max(1., y) for y in training_labels]
