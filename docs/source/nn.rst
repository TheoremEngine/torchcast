nn Submodule
============

.. autoclass:: torchcast.nn.EncoderDecoderTransformer

.. autoclass:: torchcast.nn.EncoderTransformer

.. autoclass:: torchcast.nn.NaNEncoder

.. autoclass:: torchcast.nn.TimeLastLayerNorm

Criteria
--------

We provide a collection of criteria classes that allow the target (or prediction) to be a NaN, and treat the loss on NaNs as zeros.

.. autoclass:: torchcast.nn.L1Loss

.. autoclass:: torchcast.nn.MSELoss

.. autoclass:: torchcast.nn.SmoothL1Loss

Time Embeddings
---------------

Time embeddings are used to attach the time to each token in the input. They expect to be provided both the current tokens and the times of each token.

.. autoclass:: torchcast.nn.JointEmbedding

.. autoclass:: torchcast.nn.PositionEmbedding

.. autoclass:: torchcast.nn.TimeEmbedding
