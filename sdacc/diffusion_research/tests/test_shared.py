"""Small CPU-only checks for shared refactor boundaries, not research experiments."""
import unittest,sys
import numpy as np
import torch
from unittest.mock import patch
from sdacc.diffusion_research.registry import MODELS,definition,build
from sdacc.diffusion_research.affine_math import fit,fold
from sdacc.diffusion_research.runtime import Timer
from sdacc.diffusion_research.models.flux_layers import ProjectionState

class SharedTests(unittest.TestCase):
    def test_model_registry_and_extension_boundary(self):
        self.assertEqual(definition('pixart-sigma').name,'sigma')
        self.assertEqual(definition('flux').steps,50)
        self.assertEqual(definition('sd3.5-large').name,'sd3.5-large')
        self.assertEqual(definition('sd3.5-large').units,226)
        self.assertEqual(definition('dit-xl').status,'implemented')
        self.assertEqual(definition('dit-xl').units,84)
        self.assertFalse(torch.cuda.is_initialized())
    def test_closed_form_and_bias_fold(self):
        x=np.arange(12,dtype=np.float64).reshape(6,2);y=2*x+3
        r=fit(6,np.stack([x.sum(0),y.sum(0),(x*x).sum(0),(x*y).sum(0),(y*y).sum(0)]))
        np.testing.assert_allclose(r['a'],[2,2]);np.testing.assert_allclose(r['b'],[3,3])
        sw,bias=fold(r['a'],r['b'],np.ones(2),np.array([5.,7.]));np.testing.assert_allclose(sw,[2,2]);np.testing.assert_allclose(bias,[13,17])
    def test_partial_six_group_counts_padding_before_real_columns(self):
        from sdacc.diffusion_research.reconstruction import compress,Curvature
        w=torch.arange(1,29,dtype=torch.float32).reshape(2,14)
        c=Curvature(torch.eye(14),torch.zeros(14,dtype=torch.bool),0.)
        r=compress(w,c,6,allow_tail=True)
        self.assertFalse(r['zeros'][:,-2:].any())
        self.assertEqual(int(r['zeros'].sum()),8)
        self.assertTrue((r['qweight'][:,-2:]!=0).all())
    def test_timer_encloses_select_and_call(self):
        events=[]
        class Event:
            def __init__(self,**kwargs):pass
            def record(self):events.append('event')
        with patch('torch.cuda.Event',Event):
            timer=Timer(torch.nn.Identity(),lambda x:events.append('forward') or x,2,lambda t:events.append('select'))
            timer.enabled=True;timer(torch.tensor([1]))
        self.assertEqual(events,['event','select','forward','event']);self.assertEqual(timer.index,1)
    def test_flux_protected_attention_is_not_scaled(self):
        layer=torch.nn.Linear(4,2,bias=True,dtype=torch.float32)
        with torch.no_grad():layer.weight.copy_(torch.arange(8).reshape(2,4));layer.bias.copy_(torch.tensor([2.,3.]))
        def sparse(x,output_addend=None):return torch.nn.functional.linear(x,layer.weight[:,2:],layer.bias)+output_addend
        state=ProjectionState([layer],{4:sparse},protected_prefix=2);state.select(4);x=torch.tensor([[1.,2.,3.,4.]])
        torch.testing.assert_close(state.output(x),layer(x));torch.testing.assert_close(state.teacher(x),torch.nn.functional.linear(x[:,2:],layer.weight[:,2:],layer.bias))
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
