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
import time
from typing import Any, Callable, Dict, Optional

from ..core import Tensor
from ..nn import Module


class LossFunction:
    @staticmethod
    def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Mean Squared Error Loss"""
        return ((pred - target) ** 2).mean()

    @staticmethod
    def cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Cross Entropy Loss"""
        log_softmax = pred.log_softmax(dim=1)
        return -(target * log_softmax).sum(dim=1).mean()

    @staticmethod
    def bce_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Binary Cross Entropy Loss"""
        return -(target * pred.log() + (1 - target) * (1 - pred).log()).mean()

    @staticmethod
    def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
        """L1 Loss"""
        return (pred - target).abs().mean()


class Trainer:
    def __init__(
        self,
        model: Module,
        optimizer: Any,  # Will be implemented in optimizer.py
        loss_fn: str = "cross_entropy",
        device: str = "cpu",
        callbacks: Optional[Dict[str, Callable]] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks or {}

        # Set loss function
        if isinstance(loss_fn, str):
            if loss_fn == "mse":
                self.loss_fn = LossFunction.mse_loss
            elif loss_fn == "cross_entropy":
                self.loss_fn = LossFunction.cross_entropy_loss
            elif loss_fn == "bce":
                self.loss_fn = LossFunction.bce_loss
            elif loss_fn == "l1":
                self.loss_fn = LossFunction.l1_loss
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")
        else:
            self.loss_fn = loss_fn

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move batch to device
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def eval_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Single evaluation step"""
        self.model.eval()

        with Tensor.no_grad():
            # Move batch to device
            inputs = batch["inputs"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Calculate accuracy
            predictions = outputs.argmax(dim=1)
            correct = (predictions == targets.argmax(dim=1)).sum()
            accuracy = correct.item() / targets.shape[0]

            return {"loss": loss.item(), "accuracy": accuracy}

    def fit(
        self,
        train_loader: Any,  # Will be implemented in data.py
        valid_loader: Optional[Any] = None,
        epochs: int = 10,
        log_interval: int = 100,
    ):
        """Train the model"""
        for epoch in range(epochs):
            start_time = time.time()
            train_metrics = []

            # Training loop
            for i, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                train_metrics.append(metrics)

                if i % log_interval == 0:
                    metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                    print(
                        f"Epoch {epoch + 1}/{epochs} "
                        f"[{i}/{len(train_loader)}] {metrics_str}"
                    )

            # Calculate average training metrics
            train_avg_metrics = {}
            for key in train_metrics[0].keys():
                train_avg_metrics[key] = sum(m[key] for m in train_metrics) / len(
                    train_metrics
                )

            # Validation loop
            if valid_loader is not None:
                valid_metrics = []
                for batch in valid_loader:
                    metrics = self.eval_step(batch)
                    valid_metrics.append(metrics)

                # Calculate average validation metrics
                valid_avg_metrics = {}
                for key in valid_metrics[0].keys():
                    valid_avg_metrics[key] = sum(m[key] for m in valid_metrics) / len(
                        valid_metrics
                    )

            # Log epoch metrics
            epoch_time = time.time() - start_time
            metrics_str = ", ".join(
                f"train_{k}: {v:.4f}" for k, v in train_avg_metrics.items()
            )
            if valid_loader is not None:
                metrics_str += ", " + ", ".join(
                    f"valid_{k}: {v:.4f}" for k, v in valid_avg_metrics.items()
                )
            print(
                f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s. "
                f"Metrics: {metrics_str}"
            )

            # Call callbacks
            if "on_epoch_end" in self.callbacks:
                self.callbacks["on_epoch_end"](
                    epoch=epoch,
                    metrics={
                        "train": train_avg_metrics,
                        "valid": valid_avg_metrics if valid_loader else None,
                    },
                )

    def evaluate(self, test_loader: Any) -> Dict[str, float]:
        """Evaluate the model"""
        test_metrics = []
        for batch in test_loader:
            metrics = self.eval_step(batch)
            test_metrics.append(metrics)

        # Calculate average test metrics
        test_avg_metrics = {}
        for key in test_metrics[0].keys():
            test_avg_metrics[key] = sum(m[key] for m in test_metrics) / len(
                test_metrics
            )

        return test_avg_metrics
