import unittest
import torch
import torchcast as tc


class CriteriaTest(unittest.TestCase):
    def test_l1_loss(self):
        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([[float('nan')], [0.]])
        loss = tc.nn.L1Loss()(x_1, x_2).item()
        self.assertEqual(loss, 2.5)
        loss = tc.nn.l1_loss(x_1, x_2).item()
        self.assertEqual(loss, 2.5)
        # Test sum
        loss = tc.nn.l1_loss(x_1, x_2, reduction='sum').item()
        self.assertEqual(loss, 5.0)
        # Test handling of all NaNs
        x_2 = torch.tensor([[float('nan')], [float('nan')]])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.l1_loss(x_1, x_2)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue((x_1.grad == 0).all())
        self.assertTrue((x_2.grad == 0).all())

    def test_mse_loss(self):
        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([[float('nan')], [0.]])
        loss = tc.nn.MSELoss()(x_1, x_2).item()
        self.assertEqual(loss, 6.5)
        loss = tc.nn.mse_loss(x_1, x_2).item()
        self.assertEqual(loss, 6.5)
        # Test sum
        loss = tc.nn.mse_loss(x_1, x_2, reduction='sum').item()
        self.assertEqual(loss, 13.0)
        # Test handling of all NaNs
        x_2 = torch.tensor([[float('nan')], [float('nan')]])
        x_1.requires_grad = x_2.requires_grad = True
        x_1.retain_grad(), x_2.retain_grad()
        loss = tc.nn.l1_loss(x_1, x_2)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue((x_1.grad == 0).all())
        self.assertTrue((x_2.grad == 0).all())

    def test_soft_l1_loss(self):
        pred = torch.tensor([1., 2., 0.5, 2, -2])
        target = torch.tensor([1, 1, 0.25, 0.5, 0])
        loss = tc.nn.SoftL1Loss((0, 1))(pred, target)
        self.assertEqual(loss, 1.75 / 5)
        loss = tc.nn.soft_l1_loss(pred, target, (0, 1))
        self.assertEqual(loss, 1.75 / 5)

    def test_soft_mse_loss(self):
        pred = torch.tensor([1., 2., 0.5, 2, -2])
        target = torch.tensor([1, 1, 0.25, 0.5, 0])
        loss = tc.nn.SoftMSELoss((0, 1))(pred, target)
        self.assertEqual(loss, (0.25**2 + 1.5**2) / 5)
        loss = tc.nn.soft_mse_loss(pred, target, (0, 1))


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

        embed = tc.nn.TimeEmbedding(32, ['H', 'Y', 'W', 'D'])
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
        net = net.cuda()
        x = torch.randn((5, 3, 8), device='cuda')
        out = net(x, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device='cuda')
        out = net(x, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 6), dtype=torch.int64, device='cuda')
        with self.assertRaises(ValueError):
            out = net(x, t_in=t_in, t_out=7)
        t_out = torch.zeros((5, 1, 7), dtype=torch.int64, device='cuda')
        out = net(x, t_out=t_out)
        self.assertEqual(out.shape, (5, 4, 7))

        # Second, with exogenous
        net = tc.nn.EncoderDecoderTransformer(
            3, 32, 4, 2, 1, exogenous_dim=7, one_hot_encode_nan_inputs=True,
        )
        net = net.cuda()
        x_in = torch.randn((5, 3, 8), device='cuda')
        exog_in = torch.randn((5, 7, 8), device='cuda')
        exog_out = torch.randn((5, 7, 7), device='cuda')
        out = net(x_in, exog_out=exog_out, exog_in=exog_in)
        self.assertEqual(out.shape, (5, 4, 7))
        out = net(x_in, exog_out=exog_out, t_out=7, exog_in=exog_in)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device='cuda')
        out = net(x_in, exog_out=exog_out, exog_in=exog_in, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        with self.assertRaises(ValueError):
            out = net(x, t_in=t_in, t_out=7)

    def test_encoder_transformer(self):
        # Smoke test only

        net = tc.nn.EncoderTransformer(
            3, 32, 4, 2, num_classes=10, one_hot_encode_nan_inputs=True,
        )
        net = net.cuda()
        x = torch.randn((5, 3, 8), device='cuda')
        cls_pred, out = net(x, t_out=7)
        self.assertEqual(cls_pred.shape, (5, 10))
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device='cuda')
        _, out = net(x, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 6), dtype=torch.int64, device='cuda')
        with self.assertRaises(ValueError):
            net(x, t_in=t_in, t_out=7)
        t_out = torch.zeros((5, 1, 7), dtype=torch.int64, device='cuda')
        _, out = net(x, t_out=t_out)
        self.assertEqual(out.shape, (5, 4, 7))

        # Second, with exogenous
        net = tc.nn.EncoderTransformer(
            3, 32, 4, 2, exogenous_dim=7, one_hot_encode_nan_inputs=True,
        )
        net = net.cuda()
        x_in = torch.randn((5, 3, 8), device='cuda')
        exog_in = torch.randn((5, 7, 8), device='cuda')
        exog_out = torch.randn((5, 7, 7), device='cuda')
        out = net(x_in, exog_out=exog_out, exog_in=exog_in)
        self.assertEqual(out.shape, (5, 4, 7))
        out = net(x_in, exog_out=exog_out, exog_in=exog_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        t_in = torch.zeros((5, 1, 8), dtype=torch.int64, device='cuda')
        out = net(x_in, exog_out=exog_out, exog_in=exog_in, t_in=t_in, t_out=7)
        self.assertEqual(out.shape, (5, 4, 7))
        with self.assertRaises(ValueError):
            out = net(x, t_in=t_in, t_out=7)


if __name__ == '__main__':
    unittest.main()
