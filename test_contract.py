import unittest
import numpy as np
import torch
from model import DG_MMT
from preprocessing import AUSConfig, SignalProcessor, window_count, normalize_window, process_aus_frame


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(19)

        self.cfg = AUSConfig(0., 1e-6, 50, 950, .4, 0.)

    def test_counts(self):
        self.assertEqual([window_count(x) for x in [8,5,15]], [157,97,297])
        for seconds, count in [(8,157),(5,97),(15,297)]:
            n=seconds*20
            s,a=SignalProcessor(self.cfg).process_segment(
                self.rng.normal(size=(n*50,4)), self.rng.normal(size=(n,4,1000)),
                role='test')
            self.assertEqual(s.shape,(count,4,4,50))
            self.assertEqual(a.shape,(count,4,4,128))
            np.testing.assert_allclose(s.transpose(0,1,3,2).reshape(count,200,4).mean(1),0,atol=1e-6)

    def test_no_future_information(self):
        raw=self.rng.normal(size=(500,4)); aus=self.rng.normal(size=(10,4,1000)); y=np.zeros(10,dtype=int)
        proc=SignalProcessor(self.cfg)
        a=proc.process_segment(raw,aus,role='test')
        raw[200:]=1e5;aus[4:]=1e5
        b=proc.process_segment(raw,aus,role='test')
        np.testing.assert_array_equal(a[0][0],b[0][0])
        np.testing.assert_array_equal(a[1][0],b[1][0])

    def test_boundaries(self):
        with self.assertRaises(ValueError):SignalProcessor(self.cfg).process_segment([],[],role='monitoring')
        with self.assertRaises(ValueError):SignalProcessor(self.cfg).process_segment(np.zeros((201,4)),np.zeros((4,4,1000)),role='test')

    def test_signal_equations(self):
        np.testing.assert_array_equal(normalize_window(np.ones((200,4))),np.zeros((200,4)))
        raw=np.ones((4,1000))*2
        expected=np.log1p(.4*2)/np.log1p(.4)
        np.testing.assert_allclose(process_aus_frame(raw,self.cfg),expected,rtol=1e-6)

    def test_gate_and_model(self):
        torch.manual_seed(4);m=DG_MMT().eval()
        torch.manual_seed(4);base=DG_MMT(dynamic_gate=False).eval()
        self.assertEqual(sum(p.numel() for p in m.parameters())-sum(p.numel() for p in base.parameters()),129)
        for name,value in base.state_dict().items():torch.testing.assert_close(value,m.state_dict()[name])
        self.assertTrue(all(layer.norm_first for layer in m.transformer.layers))
        self.assertTrue(all(layer.dropout.p == 0.0 for layer in m.transformer.layers))
        self.assertEqual([type(layer) for layer in m.classifier],
                         [torch.nn.Flatten, torch.nn.Linear, torch.nn.Dropout, torch.nn.Linear])
        self.assertEqual(sum(isinstance(x,torch.nn.BatchNorm1d) for x in m.modules()),4)
        x=torch.randn(1,4,4,50);a=torch.randn(1,4,4,128)
        logits,g=m(x,a,return_gating=True)
        self.assertEqual(logits.shape,(1,9));self.assertEqual(g.shape,(1,1))
        self.assertTrue(((g>0)&(g<1)).all())
        captures={}
        def hook(_module,args):captures['fused']=args[0].detach()
        handle=m.transformer.register_forward_pre_hook(hook)
        with torch.no_grad():
            m.gating_network[0].weight.zero_();m.gating_network[0].bias.zero_()
            m(x,a)
            es=torch.stack([m.semg_proj(m.semg_conv(x[:,i]).squeeze(-1)) for i in range(4)],1)
            ea=torch.stack([m.us_proj(m.us_conv(a[:,i]).squeeze(-1)) for i in range(4)],1)
            torch.testing.assert_close(captures['fused'],torch.cat((.5*es,.5*ea),1)+m.pos_embed)
        handle.remove()
        torch.testing.assert_close(m.predict_proba(x,a).sum(-1),torch.ones(1))
        m.train();m(x,a).sum().backward()
        self.assertIsNotNone(m.gating_network[0].weight.grad)
        self.assertEqual(DG_MMT(3).eval()(x,a).shape,(1,3))
        with self.assertRaises(ValueError):m(x[:,:,:,:49],a)

    def test_streaming_state_and_independent_segments(self):
        raw=self.rng.normal(size=(1000,4))
        proc=SignalProcessor(self.cfg)
        whole=proc.filter_semg_chunk(raw)
        proc.reset_semg_state()
        chunks=np.concatenate([proc.filter_semg_chunk(raw[:17]),
                               proc.filter_semg_chunk(raw[17:203]),
                               proc.filter_semg_chunk(raw[203:])])
        np.testing.assert_allclose(whole,chunks,rtol=1e-12,atol=1e-12)
        aus=self.rng.normal(size=(20,4,1000))
        first=proc.process_segment(raw,aus,role='test')
        proc.filter_semg_chunk(raw*100)
        second=proc.process_segment(raw,aus,role='test')
        np.testing.assert_array_equal(first[0],second[0])

    def test_reference_configuration(self):
        m=DG_MMT(conv_channels=(8,24),ffn_dim=96).eval()
        self.assertEqual(m(torch.zeros(1,4,4,50),torch.zeros(1,4,4,128)).shape,(1,9))
        with self.assertRaises(ValueError):DG_MMT(conv_channels=(0,32))
        with self.assertRaises(ValueError):DG_MMT(classifier_dropout=.1)


if __name__=='__main__':unittest.main(verbosity=2)
