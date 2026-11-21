import unittest
import torch
from recurrent_state_codec import BlockLayout, pack, unpack


class CodecTests(unittest.TestCase):
    def test_formats_and_bytes(self):
        for target in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            src = torch.linspace(-1.375, 1.0, 256).reshape(2,8,16).to(target)
            for modes in ([0]*8, [1]*8, [2]*8, [0,1,2,1,0,1,1,0]):
                with self.subTest(target=target,modes=modes):
                    layout = BlockLayout.from_modes(modes,target)
                    saved = pack(src,layout)
                    restored = unpack(saved)
                    self.assertEqual(saved.payload_bytes,2*sum([20,23,32][m] for m in modes))
                    self.assertTrue(torch.isfinite(restored).all())
                    for i,m in enumerate(modes):
                        row=src[:,i]
                        if m==2:
                            expected=row.half().float()
                        else:
                            scale=row.abs().amax(-1,keepdim=True)/127
                            expected=(row/scale).round()*scale
                            if m==1:
                                error=row-expected
                                index=error.abs().argmax(-1,keepdim=True)
                                expected.scatter_add_(-1,index,error.gather(-1,index).half().float())
                        torch.testing.assert_close(restored[:,i],expected,rtol=0,atol=0)

    def test_equal_budget_and_leading_dimensions(self):
        src=torch.randn(4,2,8,32)[...,::2]
        fixed=BlockLayout.uniform(8)
        mixed=BlockLayout.balanced(8,seed=17)
        self.assertEqual(pack(src,fixed).payload_bytes,pack(src,mixed).payload_bytes)
        self.assertEqual(unpack(pack(src,mixed)).shape,src.shape)
        torch.testing.assert_close(unpack(pack(torch.zeros_like(src),fixed)),torch.zeros_like(src))
        torch.testing.assert_close(mixed.modes,BlockLayout.balanced(8,seed=17).modes)

    def test_independent_batches(self):
        state=torch.randn(2,8,16)
        layout=BlockLayout.balanced(8)
        together=unpack(pack(state,layout))
        for i in range(2):
            torch.testing.assert_close(unpack(pack(state[i],layout)),together[i],rtol=0,atol=0)

    def test_invalid_inputs(self):
        for modes in ([], [[0,1]], [0,3], [0.5,1]):
            with self.assertRaises(ValueError):
                BlockLayout.from_modes(modes)
        with self.assertRaises(ValueError):
            BlockLayout.balanced(3)
        layout=BlockLayout.uniform(8)
        for src in (torch.zeros(8,17),torch.zeros(8,16).half()):
            with self.assertRaises(ValueError):
                pack(src,layout)

    def test_ties(self):
        row=torch.ones(1,16)
        row[0,0]=127
        row[0,4]=row[0,8]=0.5
        saved=pack(row,BlockLayout.uniform(1))
        self.assertEqual(saved.indices.item(),4)


if __name__=='__main__':
    unittest.main()
