# `dataloader`

## `_collate_fn`

    Convert a list of samples to a batch

## `_scan_dir`

    Scan directory and build dataset index

## `_is_image_file`

    Check if a file is an image

## `__getitem__`

    Args:
        index (int): Index

    Returns:
        Dict containing:
            'inputs': Tensor image
            'targets': Class label

