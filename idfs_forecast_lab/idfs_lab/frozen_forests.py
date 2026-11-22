"""Pure array export/inference for frozen scalar regression forests; no fitting."""
from pathlib import Path

import numpy as np


ARRAY_NAMES = ('feature', 'threshold', 'left', 'right', 'value', 'roots',
               'tree_model', 'feature_counts', 'tree_counts', 'max_depth', 'version')


class FrozenForests:
    """Finite-input scalar forests, in original model and estimator order.

    Only known, finite R1 inputs are supported. NaN-routing, classification,
    multi-output trees and fitted preprocessing are deliberately unsupported.
    """

    def __init__(self, arrays):
        if set(arrays) != set(ARRAY_NAMES):
            raise ValueError('complete frozen forest arrays required')
        self.arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
        a = self.arrays
        if any(v.ndim != 1 for v in a.values()):
            raise ValueError('one-dimensional typed arrays required')
        if any(a[k].dtype.kind not in 'iu' for k in
               ('feature', 'left', 'right', 'roots', 'tree_model', 'feature_counts',
                'tree_counts', 'max_depth', 'version')):
            raise ValueError('integer index arrays required')
        if any(a[k].dtype != np.dtype('float64') for k in ('threshold', 'value')):
            raise ValueError('float64 thresholds and values required')
        if a['version'].tolist() != [1] or a['max_depth'].size != 1:
            raise ValueError('known format/depth required')
        n = len(a['feature']); nt = len(a['roots']); nm = len(a['feature_counts'])
        if (not 0 < n <= 1_000_000 or not 0 < nt <= 10_000 or not 0 < nm <= 100
                or any(len(a[k]) != n for k in ('threshold','left','right','value'))
                or len(a['tree_model']) != nt or len(a['tree_counts']) != nm
                or not 0 <= int(a['max_depth'][0]) <= 64
                or np.any(a['feature_counts'] <= 0) or np.any(a['feature_counts'] > 100_000)
                or np.any(a['tree_counts'] <= 0) or np.any(a['tree_counts'] > nt)
                or int(a['tree_counts'].sum()) != nt):
            raise ValueError('invalid forest dimensions')
        expected_models = np.repeat(np.arange(nm), a['tree_counts'].astype(np.int64))
        if not np.array_equal(a['tree_model'], expected_models):
            raise ValueError('original contiguous model/tree order required')
        if a['roots'][0] != 0 or np.any(np.diff(a['roots'].astype(np.int64)) <= 0) or a['roots'][-1] >= n:
            raise ValueError('ordered tree roots required')
        if not np.all(np.isfinite(a['threshold'])) or not np.all(np.isfinite(a['value'])):
            raise ValueError('finite thresholds and values required')
        if np.any(a['value'] < 0):
            raise ValueError('nonnegative regression values required')
        # Check each tree independently: forward children prohibit cycles and
        # cross-tree indexing. Every node must be reachable at the declared depth.
        ends = np.r_[a['roots'][1:], n]
        for root, end, mi in zip(a['roots'], ends, a['tree_model']):
            root, end, mi = int(root), int(end), int(mi)
            reached = set(); pending = [(root, 0)]
            while pending:
                node, depth = pending.pop()
                if node in reached or depth > int(a['max_depth'][0]):
                    raise ValueError('duplicate path or invalid tree depth')
                reached.add(node)
                left, right = int(a['left'][node]), int(a['right'][node])
                if left == right == -1:
                    if a['feature'][node] != -2:
                        raise ValueError('leaf feature sentinel differs')
                    continue
                if not node < left < end or not node < right < end or left == right:
                    raise ValueError('invalid child index')
                if not 0 <= int(a['feature'][node]) < int(a['feature_counts'][mi]):
                    raise ValueError('feature outside model schema')
                pending.extend(((right,depth+1),(left,depth+1)))
            if len(reached) != end-root:
                raise ValueError('unreachable tree nodes')
        for v in a.values():
            v.setflags(write=False)

    @classmethod
    def from_models(cls, models):
        """Copy existing sklearn-compatible tree arrays without learning values."""
        feature=[];threshold=[];left=[];right=[];value=[];roots=[];tree_model=[]
        feature_counts=[];tree_counts=[];max_depth=0
        for mi, model in enumerate(models):
            if (getattr(model,'_estimator_type',None) != 'regressor'
                    or getattr(model,'n_outputs_',None) != 1 or not model.estimators_):
                raise ValueError('frozen scalar regression estimators required')
            feature_counts.append(int(model.n_features_in_));tree_counts.append(len(model.estimators_))
            for estimator in model.estimators_:
                tree=estimator.tree_;offset=len(feature);count=int(tree.node_count)
                if np.asarray(tree.value).shape != (count,1,1):
                    raise ValueError('scalar regression tree values required')
                roots.append(offset);tree_model.append(mi)
                feature.extend(np.asarray(tree.feature).tolist())
                threshold.extend(np.asarray(tree.threshold).tolist())
                left.extend((np.asarray(tree.children_left)+offset*(np.asarray(tree.children_left)>=0)).tolist())
                right.extend((np.asarray(tree.children_right)+offset*(np.asarray(tree.children_right)>=0)).tolist())
                value.extend(np.asarray(tree.value)[:,0,0].tolist())
                max_depth=max(max_depth,int(tree.max_depth))
        raw=dict(feature=feature,threshold=threshold,left=left,right=right,value=value,
                 roots=roots,tree_model=tree_model,feature_counts=feature_counts,
                 tree_counts=tree_counts,max_depth=[max_depth],version=[1])
        return cls({k:np.asarray(v,dtype=np.float64 if k in ('threshold','value') else np.int64)
                    for k,v in raw.items()})

    def save(self, path):
        with Path(path).open('xb') as f:
            np.savez_compressed(f, **self.arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k:z[k] for k in z.files})

    def predict_row(self, vectors):
        a=self.arrays;nm=len(a['feature_counts'])
        if len(vectors) != nm:
            raise ValueError('one feature vector per frozen model required')
        inputs=np.zeros((nm,int(a['feature_counts'].max())),dtype=np.float32)
        for mi,(vector,width) in enumerate(zip(vectors,a['feature_counts'])):
            vector=np.asarray(vector,dtype=np.float32)
            if vector.shape != (int(width),) or not np.all(np.isfinite(vector)):
                raise ValueError('finite ordered model features required')
            inputs[mi,:int(width)]=vector
        nodes=a['roots'].copy()
        for _ in range(int(a['max_depth'][0])):
            active=np.flatnonzero(a['left'][nodes] != -1)
            if not len(active):
                break
            current=nodes[active]
            observed=inputs[a['tree_model'][active],a['feature'][current]]
            nodes[active]=np.where(observed <= a['threshold'][current],
                                   a['left'][current],a['right'][current])
        if np.any(a['left'][nodes] != -1):
            raise ValueError('tree failed to reach a leaf')
        return np.bincount(a['tree_model'],weights=a['value'][nodes],minlength=nm)/a['tree_counts']

    def predict_batch(self, model_matrices):
        if len(model_matrices) != len(self.arrays['feature_counts']):
            raise ValueError('one matrix per model required')
        matrices=[np.asarray(x) for x in model_matrices]
        if (any(x.ndim != 2 or x.shape[1] != int(width) for x,width in
                zip(matrices,self.arrays['feature_counts']))
                or len({len(x) for x in matrices}) != 1):
            raise ValueError('matching ordered batch dimensions required')
        if not len(matrices[0]):
            return np.empty((0,len(matrices)),dtype=np.float64)
        return np.asarray([self.predict_row([x[i] for x in matrices]) for i in range(len(matrices[0]))])
