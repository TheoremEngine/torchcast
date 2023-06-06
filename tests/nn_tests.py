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
    # TODO: Add tests for TimeEmbedding, TransformerLayer
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


class TransformerTest(unittest.TestCase):
    def test_encoder_transformer(self):
        # Smoke test only
        net = tc.nn.EncoderTransformer(
            3, 32, 4, num_classes=10, num_output_channels=3,
            one_hot_encode_nan_inputs=True
        )
        net = net.cuda()
        x = torch.randn((5, 3, 8), device='cuda')
        cls_pred, out = net(x)

        self.assertEqual(cls_pred.shape, (5, 10))
        self.assertEqual(out.shape, (5, 3, 8))


if __name__ == '__main__':
    unittest.main()
