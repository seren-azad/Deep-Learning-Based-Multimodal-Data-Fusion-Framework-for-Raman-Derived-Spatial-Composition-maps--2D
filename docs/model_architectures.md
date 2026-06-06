# Model architectures

All three models predict a tensor of shape `(6, 6, 6)` from an input image of shape `(150, 150, 1)`.

## Model 1

Basic convolutional encoder with batch normalization, dropout, global average pooling, and a dense regression head.

## Model 2

Residual encoder with a multiplicative grid-attention head. The dense regression output is multiplied by a learned attention tensor with the same grid shape.

## Model 3

Enhanced residual encoder with squeeze-and-excitation blocks, spatial dropout, an exponential learning-rate schedule, and the same grid-attention idea as Model 2.

## Output activation

The notebook used linear output activations because the task is regression. This refactor preserves that design.
