from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()

    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


def max(input: Tensor, dim: int) -> Tensor:
    return input.max(dim)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.max(dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    if ignore:
        return input
    else:
        bernoulli = rand(input.shape) > p
        return input * bernoulli


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    max_val = max(input, dim)
    stable_input = input - max_val
    exp_stable = stable_input.exp()

    return exp_stable / exp_stable.sum(dim)


def logsoftmax(input: Tensor, dim: int = -1) -> Tensor:
    # logsoftmax = x - log-sum-exp
    max_val = max(input, dim)
    stable_input = input - max_val
    logsumexp = max_val + stable_input.exp().sum(dim).log()

    return input - logsumexp


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Returns a tensor of the same size as input with 1.0 in the position of the
    maximum value along dim and 0.0 otherwise.
    """
    # Get the max values along dimension
    max_vals = input.max(dim)
    # Create a mask that's 1 where input equals max_vals
    return input == max_vals
