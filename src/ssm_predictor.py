"""
A small structured state-space neural network (S4D-style) that predicts a
moving obstacle's future trajectory from its recently observed motion.

This is a genuinely different "state space" than the rest of the project:
vehicle_model.py / dynamic_vehicle_model.py are hand-derived physical
state-space models (Newton's laws in, dynamics out); this one is a *learned*
linear state-space layer in the S4/S4D/Mamba family -- the state-space
formulation shows up in the neural network's internal recurrence, not in the
physics. Both are literally "x_{t+1} = A x_t + B u_t, y_t = C x_t + D u_t";
the difference is where A, B, C, D come from (Newton vs. gradient descent).

Architecture (deliberately small -- this is a demonstration of the idea, not
a production trajectory-forecasting model):

  1. Encode the K observed per-step displacements through a stack of S4D
     blocks, each an independent diagonal linear SSM per channel (complex
     eigenvalues, represented as real 2x2 rotation-decay blocks -- see
     S4DLayer) with a GLU nonlinearity and residual connection, exactly the
     S4/S4D block pattern (Gu, Goel, Re, "Efficiently Modeling Long
     Sequences with Structured State Spaces," 2022; Gu, Gupta, Goel, Re,
     "On the Parameterization and Initialization of Diagonal State Space
     Models," 2022).
  2. Decode H future steps *autoregressively*: at each future step, the
     block stack's own previous-step prediction becomes its next input,
     continuing the same recurrent state forward. At inference this is a
     genuine free-running forecast -- the network extrapolating its learned
     dynamics with no new observations. During training this step uses
     teacher forcing (the true future displacement, standard for
     seq2seq training) for a more stable training signal.

Output: predicted *cumulative offsets* from the last observed position, for
each of the H future steps -- matching the (past, future) format built by
obstacle_trajectory_data.build_dataset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DLayer(nn.Module):
    """One diagonal SSM per channel: d_model independent channels, each with
    d_state complex modes. Complex eigenvalue lambda = -decay + i*omega is
    represented as a real 2x2 rotation-decay acting on (a, b) = (Re x, Im x)
    -- mathematically identical to a complex diagonal SSM, implemented with
    plain real tensors to avoid complex-autograd edge cases."""

    def __init__(self, d_model: int, d_state: int, dt: float):
        super().__init__()
        self.d_model, self.d_state, self.dt = d_model, d_state, dt
        # Decay: softplus(log_decay) > 0 always, so Re(lambda) = -decay < 0
        # (stability). omega spans a range of oscillation frequencies at
        # init (S4D-Lin-style), so different modes specialize to different
        # timescales/frequencies during training.
        self.log_decay = nn.Parameter(torch.rand(d_model, d_state) * 2 - 2)
        self.omega = nn.Parameter(torch.rand(d_model, d_state) * (math.pi / dt))
        self.B = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.zeros(d_model))

    def init_state(self, batch_size: int, device):
        a = torch.zeros(batch_size, self.d_model, self.d_state, device=device)
        b = torch.zeros(batch_size, self.d_model, self.d_state, device=device)
        return (a, b)

    def step(self, u: torch.Tensor, state):
        """u: (B, d_model) input at this timestep. Returns (y, new_state)."""
        a, b = state
        decay = F.softplus(self.log_decay) + 1e-3
        r = torch.exp(-decay * self.dt)
        theta = self.omega * self.dt
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        drive = self.dt * self.B * u.unsqueeze(-1)   # (B, d_model, d_state)
        a_new = r * (cos_t * a - sin_t * b) + drive
        b_new = r * (sin_t * a + cos_t * b)

        y = (self.C_re * a_new - self.C_im * b_new).sum(-1) + self.D * u
        return y, (a_new, b_new)


class S4DBlock(nn.Module):
    """Pre-norm S4D layer + GLU + residual, the standard S4/S4D block."""

    def __init__(self, d_model: int, d_state: int, dt: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, d_state, dt)
        self.gate = nn.Linear(d_model, 2 * d_model)

    def init_state(self, batch_size, device):
        return self.ssm.init_state(batch_size, device)

    def step(self, u: torch.Tensor, state):
        residual = u
        x = self.norm(u)
        y, state = self.ssm.step(x, state)
        g = self.gate(y)
        y1, y2 = g.chunk(2, dim=-1)
        y = y1 * torch.sigmoid(y2)
        return residual + y, state


class ObstaclePredictor(nn.Module):
    def __init__(self, K: int = 10, H: int = 10, d_model: int = 64,
                 d_state: int = 16, n_layers: int = 2, dt: float = 0.1):
        super().__init__()
        self.K, self.H, self.dt = K, H, dt
        self.input_proj = nn.Linear(2, d_model)
        self.blocks = nn.ModuleList([S4DBlock(d_model, d_state, dt) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, past_disp: torch.Tensor, future_disp_teacher: torch.Tensor | None = None,
                teacher_forcing_prob: float = 1.0):
        """past_disp: (B, K, 2) observed per-step displacements.
        future_disp_teacher: (B, H, 2) true per-step displacements, used for
        teacher forcing during training (omit at inference for a genuine
        autoregressive/free-running forecast).
        teacher_forcing_prob: per-sample, per-step probability of feeding the
        *true* previous displacement instead of the model's own last
        prediction (only relevant when future_disp_teacher is given).
        Training with teacher_forcing_prob=1.0 throughout is the textbook
        seq2seq setup, but it creates a train/inference mismatch ("exposure
        bias," Bengio et al., "Scheduled Sampling," 2015): the model only
        ever practices predicting from *correct* history, never from its own
        (imperfect) predictions the way it must at inference. In this
        project that mismatch was not just theoretical -- training without
        scheduled sampling produced a model whose *validation* loss (always
        evaluated autoregressively, since that's the real task) got *worse*
        over training even as its teacher-forced training loss kept
        improving smoothly. Annealing teacher_forcing_prob from 1.0 toward
        0.0 over training (see train_predictor.py) fixed it.
        Returns (pred_offsets, pred_disp): both (B, H, 2); pred_offsets is
        the cumulative offset from the last observed position (what the
        dataset's `future` field represents)."""
        B = past_disp.shape[0]
        device = past_disp.device
        states = [blk.init_state(B, device) for blk in self.blocks]

        for t in range(self.K):
            u = self.input_proj(past_disp[:, t, :])
            for i, blk in enumerate(self.blocks):
                u, states[i] = blk.step(u, states[i])

        prev_disp = past_disp[:, -1, :]
        preds = []
        for h in range(self.H):
            u = self.input_proj(prev_disp)
            for i, blk in enumerate(self.blocks):
                u, states[i] = blk.step(u, states[i])
            pred_disp = self.output_proj(u)
            preds.append(pred_disp)
            if future_disp_teacher is not None:
                if teacher_forcing_prob >= 1.0:
                    prev_disp = future_disp_teacher[:, h, :]
                elif teacher_forcing_prob <= 0.0:
                    prev_disp = pred_disp
                else:
                    use_teacher = (torch.rand(B, 1, device=device) < teacher_forcing_prob).float()
                    prev_disp = use_teacher * future_disp_teacher[:, h, :] + (1 - use_teacher) * pred_disp.detach()
            else:
                prev_disp = pred_disp

        pred_disp = torch.stack(preds, dim=1)
        pred_offsets = torch.cumsum(pred_disp, dim=1)
        return pred_offsets, pred_disp


def offsets_to_per_step_disp(offsets: torch.Tensor) -> torch.Tensor:
    """(B, H, 2) cumulative offsets -> (B, H, 2) per-step displacements."""
    first = offsets[:, :1, :]
    rest = offsets[:, 1:, :] - offsets[:, :-1, :]
    return torch.cat([first, rest], dim=1)
