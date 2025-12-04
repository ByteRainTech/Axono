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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..core import Tensor
from ..nn import Module


class ModelVisualizer:
    """Visualize model architecture and computational graph"""

    def __init__(self, model: Module):
        self.model = model
        self.graph = {}
        self._build_graph()

    def _build_graph(self):
        """Build computational graph from model"""

        def _add_module(module: Module, name: str = ""):
            for child_name, child in module.named_children():
                child_full_name = f"{name}.{child_name}" if name else child_name
                self.graph[child_full_name] = {
                    "type": type(child).__name__,
                    "params": {
                        name: tensor.shape for name, tensor in child.named_parameters()
                    },
                    "children": [],
                }
                _add_module(child, child_full_name)

        _add_module(self.model)

    def plot(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot model architecture"""
        import networkx as nx

        graph = nx.DiGraph()
        pos = {}
        labels = {}

        # Add nodes and edges
        y_offset = 0
        for name, info in self.graph.items():
            graph.add_node(name)
            pos[name] = (len(name.split(".")), y_offset)
            labels[name] = f"{info['type']}\n{name}"
            y_offset += 1

            # Add edges between parent and child modules
            parent = ".".join(name.split(".")[:-1])
            if parent in self.graph:
                graph.add_edge(parent, name)

        plt.figure(figsize=figsize)
        nx.draw(
            graph,
            pos,
            labels=labels,
            with_labels=True,
            node_color="lightblue",
            node_size=2000,
            font_size=8,
            font_weight="bold",
        )
        plt.title("Model Architecture")
        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        """Print model summary"""
        total_params = 0
        trainable_params = 0

        print("Model Summary:")
        print("=" * 80)
        print(f"{'Layer':<40} {'Output Shape':<20} {'Param #':<10}")
        print("-" * 80)

        for name, module in self.model.named_modules():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if params > 0:
                print(f"{name:<40} {str(module):<20} {params:<10,d}")
                total_params += params
                trainable_params += trainable

        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")


class TrainingVisualizer:
    """Visualize training progress and metrics"""

    def __init__(self):
        self.history = {
            "train": {"loss": [], "accuracy": []},
            "valid": {"loss": [], "accuracy": []},
        }
        self.current_epoch = 0

    def update(self, metrics: Dict[str, Dict[str, float]], epoch: int) -> None:
        """Update training history with new metrics"""
        self.current_epoch = epoch

        for split in ["train", "valid"]:
            if split in metrics:
                for metric, value in metrics[split].items():
                    self.history[split][metric].append(value)

    def plot_metrics(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        epochs = range(1, self.current_epoch + 2)

        # Plot loss
        ax1.plot(epochs, self.history["train"]["loss"], "b-", label="Training")
        if self.history["valid"]["loss"]:
            ax1.plot(epochs, self.history["valid"]["loss"], "r-", label="Validation")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(epochs, self.history["train"]["accuracy"], "b-", label="Training")
        if self.history["valid"]["accuracy"]:
            ax2.plot(
                epochs, self.history["valid"]["accuracy"], "r-", label="Validation"
            )
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


class FeatureVisualizer:
    """Visualize model's feature maps and filters"""

    def __init__(self, model: Module):
        self.model = model
        self.hooks = []
        self.feature_maps = {}

    def _hook_fn(self, name: str):
        def hook(module, input, output):
            self.feature_maps[name] = output.to_numpy()

        return hook

    def register_hooks(self, layer_names: List[str]) -> None:
        """Register forward hooks for specified layers"""
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def plot_feature_maps(
        self,
        input_tensor: Tensor,
        layer_name: str,
        num_features: int = 16,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """Plot feature maps for a specific layer"""
        # Forward pass to get feature maps
        _ = self.model(input_tensor)

        if layer_name not in self.feature_maps:
            raise ValueError(f"No feature maps found for layer {layer_name}")

        feature_maps = self.feature_maps[layer_name][0]  # First batch only
        num_features = min(num_features, feature_maps.shape[0])

        # Plot feature maps
        fig, axes = plt.subplots(4, num_features // 4, figsize=figsize)
        axes = axes.ravel()

        for i in range(num_features):
            axes[i].imshow(feature_maps[i], cmap="viridis")
            axes[i].axis("off")

        plt.suptitle(f"Feature Maps - {layer_name}")
        plt.tight_layout()
        plt.show()

    def plot_filters(
        self, layer_name: str, num_filters: int = 16, figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """Plot convolutional filters for a specific layer"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                if not hasattr(module, "weight"):
                    raise ValueError(f"Layer {layer_name} has no weights")

                weights = module.weight.to_numpy()
                num_filters = min(num_filters, weights.shape[0])

                # Plot filters
                fig, axes = plt.subplots(4, num_filters // 4, figsize=figsize)
                axes = axes.ravel()

                for i in range(num_filters):
                    # For RGB filters, take mean across channels
                    if weights.shape[1] == 3:
                        filt = np.mean(weights[i], axis=0)
                    else:
                        filt = weights[i, 0]

                    axes[i].imshow(filt, cmap="viridis")
                    axes[i].axis("off")

                plt.suptitle(f"Convolution Filters - {layer_name}")
                plt.tight_layout()
                plt.show()
                return

        raise ValueError(f"Layer {layer_name} not found in model")
