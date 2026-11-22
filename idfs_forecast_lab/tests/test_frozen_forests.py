"""Hand-authored trees only: no sklearn dependency, optimizers or fitting."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'NumPy not available')
class FrozenForestTests(unittest.TestCase):
    def setUp(self):
        from idfs_lab.frozen_forests import FrozenForests
        self.runtime=FrozenForests

    def tree(self, feature=0, threshold=.1, leaves=(2.,8.)):
        return NS(tree_=NS(node_count=3,max_depth=1,
            feature=np.array([feature,-2,-2]),threshold=np.array([threshold,-2.,-2.]),
            children_left=np.array([1,-1,-1]),children_right=np.array([2,-1,-1]),
            value=np.array([0.,*leaves]).reshape(3,1,1)))

    def leaf(self, value):
        return NS(tree_=NS(node_count=1,max_depth=0,feature=np.array([-2]),threshold=np.array([-2.]),
            children_left=np.array([-1]),children_right=np.array([-1]),value=np.array([[[value]]])))

    def models(self):
        return [NS(_estimator_type='regressor',n_outputs_=1,n_features_in_=1,estimators_=[self.tree(),self.leaf(4.)]),
                NS(_estimator_type='regressor',n_outputs_=1,n_features_in_=2,estimators_=[self.tree(feature=1,threshold=1.,leaves=(10.,20.))])]

    def test_multiple_models_root_leaf_offsets_and_order(self):
        r=self.runtime.from_models(self.models())
        np.testing.assert_array_equal(r.predict_row([[0.],[-999.,2.]]),[3.,20.])
        np.testing.assert_array_equal(r.predict_row([[1.],[999.,0.]]),[6.,10.])
        np.testing.assert_array_equal(r.predict_row([[0.],[-999.,2.]]),[3.,20.])
        self.assertTrue(all(not a.flags.writeable for a in r.arrays.values()))

    def test_float32_conversion_and_inclusive_threshold(self):
        r=self.runtime.from_models(self.models()[:1])
        # Float64 .1 is <= threshold; sklearn's float32(.1) is just above it.
        np.testing.assert_array_equal(r.predict_row([[.1]]),[6.])
        threshold=float(np.float32(.1))
        models=self.models()[:1];models[0].estimators_[0]=self.tree(threshold=threshold)
        r=self.runtime.from_models(models)
        np.testing.assert_array_equal(r.predict_row([[threshold]]),[3.])
        above=np.nextafter(np.float32(.1),np.float32(np.inf))
        np.testing.assert_array_equal(r.predict_row([[above]]),[6.])

    def test_roundtrip_refuses_overwrite_and_batch_shape(self):
        r=self.runtime.from_models(self.models())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'forest.npz';r.save(p);loaded=self.runtime.load(p)
            with self.assertRaises(FileExistsError):r.save(p)
            got=loaded.predict_batch([[[0.],[1.]],[[2.,0.],[2.,2.]]])
            np.testing.assert_array_equal(got,[[3.,10.],[6.,20.]])
            self.assertEqual(loaded.predict_batch([np.empty((0,1)),np.empty((0,2))]).shape,(0,2))
        with self.assertRaises(ValueError):r.predict_batch([[[0.]],[[1.,2.],[3.,4.]]])
        with self.assertRaises(ValueError):r.predict_row([[0.],[1.]])
        with self.assertRaises(ValueError):r.predict_row([[float('nan')],[1.,2.]])

    def test_invalid_tree_cycle_cross_tree_schema_and_values_rejected(self):
        arrays=self.runtime.from_models(self.models()).arrays
        for key,index,value in [('left',0,0),('right',0,3),('feature',0,1),
                                ('value',1,float('nan')),('value',1,-1.)]:
            bad={k:np.array(v,copy=True) for k,v in arrays.items()};bad[key][index]=value
            with self.assertRaises(ValueError):self.runtime(bad)
        bad=dict(arrays);bad['feature']=bad['feature'].astype(float)
        with self.assertRaises(ValueError):self.runtime(bad)

    def test_multioutput_and_invalid_depth_rejected(self):
        models=self.models();models[0].n_outputs_=2
        with self.assertRaises(ValueError):self.runtime.from_models(models)
        models=self.models();models[0]._estimator_type='classifier'
        with self.assertRaises(ValueError):self.runtime.from_models(models)
        arrays={k:np.array(v,copy=True) for k,v in self.runtime.from_models(self.models()).arrays.items()}
        arrays['max_depth'][0]=0
        with self.assertRaises(ValueError):self.runtime(arrays)

    def test_public_tags_support_new_sklearn_without_legacy_type_marker(self):
        models=self.models()
        for model in models:
            del model._estimator_type
            model.__sklearn_tags__=lambda:NS(estimator_type='regressor')
        r=self.runtime.from_models(models)
        np.testing.assert_array_equal(r.predict_row([[0.],[-999.,2.]]),[3.,20.])
        models[0].__sklearn_tags__=lambda:NS(estimator_type='classifier')
        with self.assertRaises(ValueError):self.runtime.from_models(models)
