import torch

from cloudtrace_mvad.model import CloudTraceMVAD, parameter_count, approximate_linear_flops


def test_full_model_shape_and_size():
    model = CloudTraceMVAD([47, 21, 12], hidden_dim=96, latent_dim=32)
    x = [torch.randn(8, 47), torch.randn(8, 21), torch.randn(8, 12)]
    out = model(x)
    assert out["fused"].shape == (8, 32)
    assert out["gate"].shape == (8, 3)
    assert torch.allclose(out["gate"].sum(1), torch.ones(8), atol=1e-6)
    assert [r.shape for r in out["reconstructions"]] == [torch.Size([8, 47]), torch.Size([8, 21]), torch.Size([8, 12])]
    assert parameter_count(model) == 44915
    assert approximate_linear_flops(model) == 86592
