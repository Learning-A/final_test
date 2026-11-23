"""Fixed auxiliary training state; formal metric/zero mask never changes."""
import math


def quiet_state(actual_cluster,last_cluster):
    if any(not math.isfinite(v) or v<0 for v in (actual_cluster,last_cluster)):raise ValueError('invalid state values')
    return int(actual_cluster<=max(10.,.1*last_cluster))


def cluster_input(row):
    return sum([v[:112] for v in row['x']],[])+row['x'][0][112:]+row['extra'][0][:20]+sum([v[20:] for v in row['extra']],[])
