"""Observed-only input transforms and explicit log-target output parameterization."""
import numpy as np
import torch
from torch.nn import functional as F

from .placement import require_training_host
from .r2_scaling import checked_input, contract_sha, feature_contract
from .r2_loss_controls import R2SharedMLP


SIGNED = (30, 31, 32)
T3_MIN_COLUMNS = 18


def transform_contract():
    c = feature_contract()
    return dict(schema='c258_log_robust_v1', source_contract_sha256=contract_sha(c),
                nonnegative=[i for i in c['continuous'] if i not in SIGNED],
                signed=list(SIGNED), preserved=c['preserved'], names=c['names'],
                observed_by_column=c['observed_by_column'], reference_unit=1.,
                quantile_method='linear', zero_IQR_scale=1., empty_center=0., empty_scale=1.,
                t3_gate=dict(min_observed=1000, min_distinct=100, absolute_log_skew_gt=2.,
                             minimum_eligible_columns=T3_MIN_COLUMNS))


def _forward_column(values, method, lam=1.):
    if method == 'log1p':
        return np.log1p(values)
    if abs(lam) < 1e-8:
        return np.log1p(values)
    return np.expm1(lam * np.log1p(values)) / lam


def fit_input_transform(values, names, *, method, source_role, partition):
    require_training_host()
    if source_role != 'train' or partition != 'fit_run01_02' or method not in ('log1p', 'yeojohnson'):
        raise ValueError('fixed fitting partition and registered method required')
    x, c = checked_input(values, names)
    if not len(x):
        raise ValueError('nonempty fitting input required')
    contract = transform_contract()
    state = dict(contract=contract, method=method, source_role=source_role, partition=partition,
                 center=[0.]*258, scale=[1.]*258, observed_count=[0]*258,
                 lambdas=[1.]*258, diagnostics=[], fitted_on='lmj@b507')
    for col in c['continuous']:
        mask = c['observed_by_column'].get(str(col))
        v = x[:, col] if mask is None else x[x[:, mask] > 0, col]
        state['observed_count'][col] = len(v)
        if not len(v):
            continue
        if col not in SIGNED:
            logv = np.log1p(v)
            centered = logv - logv.mean()
            sd = logv.std()
            skew = float(np.mean((centered / sd)**3)) if sd > 1e-12 else 0.
            distinct = len(np.unique(v))
            eligible = len(v) >= 1000 and distinct >= 100 and abs(skew) > 2.
            state['diagnostics'].append(dict(column=col, observed=len(v), distinct=distinct,
                                            log_skew=skew, t3_eligible=eligible))
            if method == 'yeojohnson' and distinct > 1:
                from scipy.stats import yeojohnson_normmax
                state['lambdas'][col] = float(yeojohnson_normmax(v))
            v = _forward_column(v, method, state['lambdas'][col])
        if not np.isfinite(v).all():
            raise ValueError('nonfinite fitting transform; no clipping fallback')
        q25, med, q75 = np.quantile(v, [.25, .5, .75], method='linear')
        state['center'][col] = float(med)
        state['scale'][col] = float(q75-q25) if q75 > q25 else 1.
    state['t3_eligible_columns'] = sum(r['t3_eligible'] for r in state['diagnostics'])
    state['t3_admitted'] = state['t3_eligible_columns'] >= T3_MIN_COLUMNS
    return state


def apply_input_transform(values, names, state):
    x, c = checked_input(values, names)
    if state.get('contract') != transform_contract() or state.get('method') not in ('log1p', 'yeojohnson'):
        raise ValueError('registered transform identity differs')
    if state.get('source_role') != 'train' or state.get('partition') != 'fit_run01_02':
        raise ValueError('fitting provenance differs')
    center, scale, count, lambdas = [np.asarray(state[k]) for k in ('center', 'scale', 'observed_count', 'lambdas')]
    if any(a.shape != (258,) or not np.isfinite(a).all() for a in (center, scale, count, lambdas)):
        raise ValueError('finite C258 transform state required')
    if (scale <= 0).any() or count.dtype.kind not in 'iu' or (count < 0).any():
        raise ValueError('positive scales and integer observation counts required')
    if (center[c['preserved']] != 0).any() or (scale[c['preserved']] != 1).any():
        raise ValueError('preserved columns cannot be rescaled')
    empty = [i for i in c['continuous'] if count[i] == 0]
    if (center[empty] != 0).any() or (scale[empty] != 1).any():
        raise ValueError('unobserved fitting columns require defaults')
    out = x.copy()
    for col in c['continuous']:
        v = x[:, col] if col in SIGNED else _forward_column(x[:, col], state['method'], lambdas[col])
        out[:, col] = (v-center[col])/scale[col]
        mask = c['observed_by_column'].get(str(col))
        if mask is not None:
            out[x[:, mask] == 0, col] = 0
    if not np.isfinite(out).all():
        raise ValueError('nonfinite transformed features; fail without clipping')
    return out


class LogTargetMLP(R2SharedMLP):
    """Same parameters, z=log1p(s)*softplus(h), y=expm1(z); no bias correction."""

    def forward(self, features, node_indices):
        # The inherited constructor fixes the same hidden weights and head at softplus(h)=1.
        combined = torch.cat((features, self.node_embedding(node_indices)), dim=1)
        z = F.softplus(self.backbone(combined).squeeze(-1)) * torch.log1p(self.output_scales[node_indices])
        if not bool(torch.isfinite(z).all()) or bool((z > 80).any()):
            raise ValueError('log output overflow guard; no automatic clipping')
        return torch.expm1(z)
