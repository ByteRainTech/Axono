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
import random
from typing import List, Tuple, Union

import numpy as np
from PIL import Image


class Transform:
    def __call__(self, img):
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size, Image.BILINEAR)


class RandomCrop(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = 0):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.padding > 0:
            img = pad(img, self.padding)

        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return img

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return img.crop((j, i, j + tw, i + th))


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation(Transform):
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, float):
            degrees = (-degrees, degrees)
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return img.rotate(angle, Image.BILINEAR, expand=False)


class Normalize(Transform):
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img


class ToTensor(Transform):
    def __call__(self, img: Union[Image.Image, np.ndarray]):
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Handle PIL Image
        if len(img.shape) == 2:
            img = img[:, :, None]

        # Convert HWC to CHW format
        img = img.transpose((2, 0, 1))
        return img


def pad(img: Image.Image, padding: int) -> Image.Image:
    """Helper function to pad an image"""
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])

    w, h = img.size
    new_w = w + padding[0] + padding[2]
    new_h = h + padding[1] + padding[3]

    result = Image.new(img.mode, (new_w, new_h), 0)
    result.paste(img, (padding[0], padding[1]))
    return result
