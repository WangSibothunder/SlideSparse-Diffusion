"""Small native CPU models check routing, not sparse speed or image quality."""
import unittest
import torch
from torch.nn import functional as F
from diffusers import SD3Transformer2DModel, Transformer2DModel
from sdacc.diffusion_research.models.sd3 import install_projections as install_joint
from sdacc.diffusion_research.models.dit_xl import DiTXLModel
from sdacc.diffusion_research.models.sigma import install_projections as install_pixart


def exact_operators(model, units):
    operators={}
    for unit in units:
        members=[model.get_submodule(n) for n in unit['targets']]
        w=torch.cat([m.weight for m in members]);b=torch.cat([m.bias for m in members])
        def linear(x,w=w,b=b):return F.linear(x,w,b)
        operators[unit['unit']]={4:linear}
    return operators


class ModelRoutingTests(unittest.TestCase):
    @torch.inference_mode()
    def test_sd35_native_qk_normalization_and_affine_operator_selection(self):
        torch.manual_seed(7)
        model=SD3Transformer2DModel(sample_size=4,patch_size=2,in_channels=4,out_channels=4,
            num_layers=2,attention_head_dim=8,num_attention_heads=2,joint_attention_dim=16,
            caption_projection_dim=16,pooled_projection_dim=8,pos_embed_max_size=4,
            qk_norm='rms_norm').eval()
        from sdacc.diffusion_research.models.sd3 import SD35LargeModel
        units=SD35LargeModel.inventory(model,32,3)
        args=dict(hidden_states=torch.randn(2,4,4,4),encoder_hidden_states=torch.randn(2,80,16),
                  pooled_projections=torch.randn(2,8),timestep=torch.tensor([4,4]),return_dict=False)
        reference=model(**args)[0]
        patched,states=install_joint(model,units,exact_operators(model,units))
        self.assertIsNotNone(patched.transformer_blocks[0].attn.norm_q)
        torch.testing.assert_close(patched(**args)[0],reference)
        for state in states:state.select(4)
        torch.testing.assert_close(patched(**args)[0],reference,rtol=1e-5,atol=1e-6)
        # AffinePolicy substitutes a prepared callable, not just an integer mode.
        for state in states:state.active=state.ops[4]
        torch.testing.assert_close(patched(**args)[0],reference,rtol=1e-5,atol=1e-6)
        self.assertEqual(len(units),10)
        self.assertEqual(units[0]['m'],8)
        self.assertEqual(units[3]['m'],160)
        self.assertFalse(torch.cuda.is_initialized())

    @torch.inference_mode()
    def test_dit_class_conditioning_fused_qkv_and_original_dense(self):
        torch.manual_seed(8)
        model=Transformer2DModel(sample_size=4,patch_size=2,in_channels=4,out_channels=8,
            num_layers=2,attention_head_dim=8,num_attention_heads=2,num_embeds_ada_norm=1000,
            norm_type='ada_norm_zero',norm_elementwise_affine=False,attention_bias=True,
            activation_fn='gelu-approximate').eval()
        units=DiTXLModel.inventory(model,32,0)
        args=dict(hidden_states=torch.randn(2,4,4,4),class_labels=torch.tensor([7,1000]),
                  timestep=torch.tensor([4,4]),return_dict=False)
        reference=model(**args)[0]
        patched,states=install_pixart(model,units,exact_operators(model,units))
        torch.testing.assert_close(patched(**args)[0],reference)
        for state in states:state.select(4)
        torch.testing.assert_close(patched(**args)[0],reference,rtol=1e-5,atol=1e-6)
        self.assertEqual(len(units),6)
        self.assertEqual(units[0]['targets'],['transformer_blocks.0.attn1.to_'+n for n in ('q','k','v')])
        self.assertIs(model.transformer_blocks[0].attn1.to_q.weight,
                      patched.transformer_blocks[0].attn1.to_q.original.weight)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
