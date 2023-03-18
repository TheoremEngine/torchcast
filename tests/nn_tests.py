import unittest
import torch
import torchcast as tc


class CriteriaTest(unittest.TestCase):
    def test_l1_loss(self):
        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([[float('nan')], [0.]])
        with self.assertWarns(UserWarning):
            loss = tc.nn.L1Loss()(x_1, x_2).item()
        self.assertEqual(loss, 2.5)

    def test_mse_loss(self):
        x_1 = torch.tensor([[1., 2., 3.], [float('nan'), -2., -3.]])
        x_2 = torch.tensor([[float('nan')], [0.]])
        with self.assertWarns(UserWarning):
            loss = tc.nn.MSELoss()(x_1, x_2).item()
        self.assertEqual(loss, 6.5)

    def test_soft_l1_loss(self):
        criterion = tc.nn.SoftL1Loss()
        pred = torch.tensor([1., 2., 0.5, 2, -2])
        target = torch.tensor([1, 1, 0.25, 0.5, 0])
        loss = criterion(pred, target)
        self.assertEqual(loss, 1.75 / 5)

    def test_soft_mse_loss(self):
        criterion = tc.nn.SoftMSELoss()
        pred = torch.tensor([1., 2., 0.5, 2, -2])
        target = torch.tensor([1, 1, 0.25, 0.5, 0])
        loss = criterion(pred, target)
        self.assertEqual(loss, (0.25**2 + 1.5**2) / 5)


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


class HooksTest(unittest.TestCase):
    def test_max_norm_constraint(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                weight = torch.nn.Parameter(torch.empty((1,)))
                self.register_parameter('weight', weight)

            def forward(self):
                return self.weight

        module = tc.nn.hooks.max_norm_constraint(TestModule())

        with torch.no_grad():
            module.weight_nc.fill_(0.5)

        self.assertTrue(abs(module().item() - 0.5) < 1e-5)

        with torch.no_grad():
            module.weight_nc.fill_(-2.)

        self.assertTrue(abs(module().item() + 1.) < 1e-5)


if __name__ == '__main__':
    unittest.main()
