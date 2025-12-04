# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from ..core import Tensor


class Dataset:
    def __init__(self):
        self.transform = None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def set_transform(self, transform: Callable):
        self.transform = transform


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self._indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)

        for i in range(0, len(self._indices), self.batch_size):
            batch_indices = self._indices[i : i + self.batch_size]
            batch = self._collate_fn([self.dataset[idx] for idx in batch_indices])
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """Convert a list of samples to a batch"""
        elem = batch[0]
        if isinstance(elem, dict):
            return {
                key: self._collate_fn([d[key] for d in batch])
                if isinstance(elem[key], (dict, list))
                else Tensor.stack([d[key] for d in batch])
                if isinstance(elem[key], Tensor)
                else Tensor.from_numpy(np.stack([d[key] for d in batch]))
                for key in elem
            }
        elif isinstance(elem, list):
            return [self._collate_fn([d[i] for d in batch]) for i in range(len(elem))]
        else:
            raise TypeError(f"Unsupported batch element type: {type(elem)}")


class ImageFolder(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__()
        self.root = root
        self.transform = transform

        # Scan directory for images and classes
        self._scan_dir()

    def _scan_dir(self):
        """Scan directory and build dataset index"""
        import os

        self.classes = sorted(
            [
                d
                for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
            ]
        )

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for target_class in self.classes:
            class_path = os.path.join(self.root, target_class)
            if not os.path.isdir(class_path):
                continue

            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in sorted(fnames):
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[target_class])
                        self.samples.append(item)

    def _is_image_file(self, filename: str) -> bool:
        """Check if a file is an image"""
        img_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
        )
        return filename.lower().endswith(img_extensions)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            Dict containing:
                'inputs': Tensor image
                'targets': Class label
        """
        path, target = self.samples[index]

        # Load image
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return {"inputs": img, "targets": target}

    def __len__(self) -> int:
        return len(self.samples)
