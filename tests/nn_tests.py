import unittest
import torch
import torchcast as tc


class CriteriaTest(unittest.TestCase):
    def test_l1_loss(self):
        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([
            [float('nan'), float('nan'), float('nan')],
            [0., 0., 0.]
        ])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.L1Loss()(x_1, x_2)
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        sb_1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, -0.5, -0.5]])
        self.assertTrue(torch.isclose(x_1.grad, sb_1).all())
        sb_2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
        self.assertTrue(torch.isclose(x_2.grad, sb_2).all(), x_2.grad)

        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([
            [float('nan'), float('nan'), float('nan')],
            [0., 0., 0.]
        ])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.l1_loss(x_1, x_2, reduction='sum')
        self.assertEqual(loss.item(), 5.0)
        loss.backward()
        sb_1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, -1.0, -1.0]])
        self.assertTrue(torch.isclose(x_1.grad, sb_1).all())
        sb_2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        self.assertTrue(torch.isclose(x_2.grad, sb_2).all())

        x_1 = torch.tensor([
            [1., 2., 3.],
            [float('nan'), float('nan'), float('nan')]
        ])
        x_2 = torch.tensor([
            [float('nan'), float('nan'), float('nan')],
            [0., 0., 0.]
        ])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.l1_loss(x_1, x_2)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue((x_1.grad == 0).all())
        self.assertTrue((x_2.grad == 0).all())

    def test_mse_loss(self):
        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([
            [float('nan'), float('nan'), float('nan')],
            [0., 0., 0.]
        ])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.MSELoss()(x_1, x_2)
        self.assertEqual(loss.item(), 6.5)
        loss.backward()
        sb_1 = torch.tensor([-2.0, -3.0])
        self.assertTrue(torch.isclose(x_1.grad[1, 1:], sb_1).all())
        sb_2 = torch.tensor([2.0, 3.0])
        self.assertTrue(torch.isclose(x_2.grad[1, 1:], sb_2).all())

        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([
            [float('nan'), float('nan'), float('nan')],
            [0., 0., 0.]
        ])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.mse_loss(x_1, x_2, reduction='sum')
        self.assertEqual(loss.item(), 13.0)
        loss.backward()
        sb_1 = torch.tensor([-4.0, -6.0])
        self.assertTrue(torch.isclose(x_1.grad[1, 1:], sb_1).all())
        sb_2 = torch.tensor([4.0, 6.0])
        self.assertTrue(torch.isclose(x_2.grad[1, 1:], sb_2).all())

    def test_smooth_l1_loss(self):
        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([
            [float('nan'), float('nan'), float('nan')],
            [0., 0., 0.]
        ])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.SmoothL1Loss()(x_1, x_2)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        sb_1 = torch.tensor([-0.5, -0.5])
        self.assertTrue(torch.isclose(x_1.grad[1, 1:], sb_1).all())
        sb_2 = torch.tensor([0.5, 0.5])
        self.assertTrue(torch.isclose(x_2.grad[1, 1:], sb_2).all())

        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([
            [float('nan'), float('nan'), float('nan')],
            [0., 0., 0.]
        ])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.smooth_l1_loss(x_1, x_2, reduction='sum')
        self.assertEqual(loss.item(), 4.0)
        loss.backward()
        sb_1 = torch.tensor([-1.0, -1.0])
        self.assertTrue(torch.isclose(x_1.grad[1, 1:], sb_1).all())
        sb_2 = torch.tensor([1.0, 1.0])
        self.assertTrue(torch.isclose(x_2.grad[1, 1:], sb_2).all())


class LayersTest(unittest.TestCase):
    # TODO: Add tests for TimeEmbedding
    def test_nan_encoder(self):
        x = torch.tensor([[[0.5, float('nan')], [float('nan'), -0.5]]])
        encoder = tc.nn.NaNEncoder()
        out = encoder(x)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (1, 4, 2))
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertEqual(x[0, 0, 0].item(), 0.5)
        self.assertEqual(x[0, 1, 1].item(), -0.5)
        mask = torch.tensor([[0., 1.], [1., 0.]])
        self.assertTrue((out[0, 2:, :] == mask).all())

    def test_time_embedding(self):
        # Smoke test only

        embed = tc.nn.TimeEmbedding(32, ['h', 'Y', 'W', 'D'])
        x = torch.randn((3, 32, 5))
        t = torch.arange(5, dtype=torch.int64).view(1, 1, 5)
        out = embed(x, t)
        self.assertEqual(out.shape, (3, 32, 5))


class ModelsTest(unittest.TestCase):
    def test_encoder_decoder_transformer(self):
        # Smoke test only

        # First, no exogenous
        net = tc.nn.EncoderDecoderTransformer(
            3, 32, 4, 2, 1, one_hot_encode_nan_inputs=True,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)
        x = torch.randn((5, 3, 8), device=device)
        out = net(x, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device=device)
        out = net(x, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 6), dtype=torch.int64, device=device)
        with self.assertRaises(ValueError):
            out = net(x, t_in=t_in, t_out=7)
        t_out = torch.zeros((5, 1, 7), dtype=torch.int64, device=device)
        out = net(x, t_out=t_out)
        self.assertEqual(out.shape, (5, 4, 7))

        # Second, with exogenous
        net = tc.nn.EncoderDecoderTransformer(
            3, 32, 4, 2, 1, exogenous_dim=7, one_hot_encode_nan_inputs=True,
        )
        net = net.to(device)
        x_in = torch.randn((5, 3, 8), device=device)
        exog_in = torch.randn((5, 7, 8), device=device)
        exog_out = torch.randn((5, 7, 7), device=device)
        out = net(x_in, exog_out=exog_out, exog_in=exog_in)
        self.assertEqual(out.shape, (5, 4, 7))
        out = net(x_in, exog_out=exog_out, t_out=7, exog_in=exog_in)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device=device)
        out = net(x_in, exog_out=exog_out, exog_in=exog_in, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        with self.assertRaises(ValueError):
            out = net(x, t_in=t_in, t_out=7)

    def test_encoder_transformer(self):
        # Smoke test only

        net = tc.nn.EncoderTransformer(
            3, 32, 4, 2, num_classes=10, one_hot_encode_nan_inputs=True,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)
        x = torch.randn((5, 3, 8), device=device)
        cls_pred, out = net(x, t_out=7)
        self.assertEqual(cls_pred.shape, (5, 10))
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device=device)
        _, out = net(x, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 6), dtype=torch.int64, device=device)
        with self.assertRaises(ValueError):
            net(x, t_in=t_in, t_out=7)
        t_out = torch.zeros((5, 1, 7), dtype=torch.int64, device=device)
        _, out = net(x, t_out=t_out)
        self.assertEqual(out.shape, (5, 4, 7))

        # Second, with exogenous
        net = tc.nn.EncoderTransformer(
            3, 32, 4, 2, exogenous_dim=7, one_hot_encode_nan_inputs=True,
        )
        net = net.to(device)
        x_in = torch.randn((5, 3, 8), device=device)
        exog_in = torch.randn((5, 7, 8), device=device)
        exog_out = torch.randn((5, 7, 7), device=device)
        out = net(x_in, exog_out=exog_out, exog_in=exog_in)
        self.assertEqual(out.shape, (5, 4, 7))
        out = net(x_in, exog_out=exog_out, exog_in=exog_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device=device)
        out = net(x_in, exog_out=exog_out, exog_in=exog_in, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        with self.assertRaises(ValueError):
            out = net(x, t_in=t_in, t_out=7)


if __name__ == '__main__':
    unittest.main()
