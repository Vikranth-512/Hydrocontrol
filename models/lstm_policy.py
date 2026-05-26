"""
LSTM-based control policy: sequence of sensor features → (flowrate, duration).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    """
    Input (batch, seq_len, features)
    → LSTM → Dropout → LSTM → Dense → Linear(2)

    Outputs optimal_flowrate and optimal_duration.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            actions: (batch, 2)
        """
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        # Use last timestep representation
        last = out[:, -1, :]
        h = self.relu(self.fc_hidden(last))
        return self.fc_out(h)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    def export_torchscript(self, example_input: torch.Tensor, path: str) -> None:
        """Trace model for edge deployment."""
        self.eval()
        traced = torch.jit.trace(self, example_input)
        traced.save(path)

    def export_onnx(
        self,
        example_input: torch.Tensor,
        path: str,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
    ) -> None:
        torch.onnx.export(
            self,
            example_input,
            path,
            input_names=input_names or ["sensor_sequence"],
            output_names=output_names or ["control_action"],
            dynamic_axes={
                "sensor_sequence": {0: "batch"},
                "control_action": {0: "batch"},
            },
            opset_version=17,
        )
