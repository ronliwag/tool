import torch
import torch.nn as nn
import torch.nn.functional as F


class ODConvTranspose1D(nn.Module):
    """
    Real ODConv-style multi-kernel ConvTranspose1D with lightweight gating and FiLM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        speaker_dim=192,
        emotion_dim=256,
        num_kernels: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        self.num_kernels = max(1, int(num_kernels))

        self.kernels = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    output_padding,
                    groups,
                    bias,
                    dilation,
                )
                for _ in range(self.num_kernels)
            ]
        )

        gate_in_dim = in_channels + speaker_dim + emotion_dim
        hidden = max(64, min(512, gate_in_dim))
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_kernels),
        )

        self.film = nn.Sequential(
            nn.Linear(speaker_dim + emotion_dim, out_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.speaker_dim = speaker_dim
        self.emotion_dim = emotion_dim
        self.condition = None
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Expose last gating weights (for verification/logging)
        self.last_alphas = None  # shape [K] averaged over batch

    def set_condition(self, condition: torch.Tensor):
        self.condition = condition

    def _safe_condition(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        if self.condition is not None and isinstance(self.condition, torch.Tensor):
            if self.condition.dim() == 2 and self.condition.size(0) == b:
                return self.condition
        return torch.zeros(b, self.speaker_dim + self.emotion_dim, device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.in_channels:
            if x.size(1) < self.in_channels:
                pad_c = self.in_channels - x.size(1)
                x = F.pad(x, (0, 0, 0, pad_c))
            else:
                x = x[:, : self.in_channels, :]

        try:
            pooled = self.pool(x).squeeze(-1)
            cond = self._safe_condition(x)
            gate_in = torch.cat([pooled, cond], dim=1)
            alphas = torch.softmax(self.gate(gate_in), dim=1)
        except Exception:
            alphas = x.new_ones(x.size(0), self.num_kernels) / float(self.num_kernels)

        try:
            ys = []
            for k, conv in enumerate(self.kernels):
                yk = conv(x)
                alpha_k = alphas[:, k].view(-1, 1, 1)
                ys.append(yk * alpha_k)
            y = torch.stack(ys, dim=0).sum(dim=0)
            # Record mean gating weights across batch for backend verification
            try:
                self.last_alphas = alphas.mean(dim=0).detach().cpu()
            except Exception:
                self.last_alphas = None
        except Exception:
            y = self.kernels[0](x)

        cond = self._safe_condition(x)
        try:
            film_params = self.film(cond)
            scale = film_params[:, : self.out_channels].unsqueeze(-1)
            shift = film_params[:, self.out_channels :].unsqueeze(-1)
            
            # CRITICAL FIX: Limit FiLM parameters to prevent extreme scaling
            scale = torch.clamp(scale, -0.5, 0.5)  # Limit scale to reasonable range
            shift = torch.clamp(shift, -0.5, 0.5)  # Limit shift to reasonable range
            
            y = y * (1.0 + scale) + shift
            
            # Additional safety: Clamp output to prevent extreme values
            y = torch.clamp(y, -2.0, 2.0)
        except Exception:
            pass

        return y


