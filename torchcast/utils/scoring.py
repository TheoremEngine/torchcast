import torch

__all__ = ['continuous_ranked_probability_score']


def continuous_ranked_probability_score(true_values: torch.Tensor,
                                        samples: torch.Tensor, dim: int = -1):
    '''
    Calculates the continuous ranked probability score given a set of samples:

    .. math::
        \\mbox{CRPS}(F, y) = \\int (F(z) - \\mathbb{1}(y \\leq z))^2 dz

    Which, provided the distribution is well-behaved, is equivalent to:

    .. math::
        \\mbox{CRPS}(F, y) = \\mathbb{E}|X_1 - y|
        - \\frac{1}{2}\\mathbb{E}|X_1 - X_2|

    Where :math:`X_1, X_2` are independent samples from the distribution of
    :math:`F`.

    For further details, see:

        https://cran.r-project.org/web/packages/scoringRules/vignettes/article.pdf
    '''
    if true_values.ndim != samples.ndim - 1:
        raise ValueError(
            f'True values and sample shapes do not match: {true_values.shape} '
            f'vs. {samples.shape}'
        )

    m_1 = (samples - true_values.unsqueeze(dim)).abs().mean()
    m_2 = (samples.unsqueeze(dim) - samples.unsqueeze(dim + 1)).abs().mean()
    return m_1 - (m_2 / 2.)
