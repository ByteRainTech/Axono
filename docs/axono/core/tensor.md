# `tensor`

## `Tensor`

    Python Tensor class wrapping C++ Tensor

## `__init__`

    Initialize Tensor

    Args:
        dtype: Data type of tensor
        shape: Shape of tensor, if None creates empty tensor

## `create`

    Create a new tensor

## `create_like`

    Create a tensor with same shape and dtype as another

## `from_numpy`

    Create tensor from numpy array - FIXED VERSION

## `to_numpy`

    Convert tensor to numpy array - FIXED VERSION

## `reshape`

    Reshape the tensor

## `resize`

    Resize the tensor (may reallocate memory)

## `fill_zero`

    Fill tensor with zeros

## `fill`

    Fill tensor with the specified value.

    Parameters:
    -----------
    value : int or float
        The value to fill the tensor with

    Returns:
    --------
    Tensor
        self for method chaining

    Raises:
    -------
    ValueError
        If the value is incompatible with the tensor's data type
    RuntimeError
        If the fill operation fails

## `is_same_shape`

    Check if has same shape as another tensor

## `zeros`

    Create a tensor filled with zeros

## `ones`

    Create a tensor filled with ones

## `full`

    Create a tensor filled with value

