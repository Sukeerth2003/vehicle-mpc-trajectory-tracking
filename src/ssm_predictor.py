"""
A structured state-space neural network that predicts a moving obstacle's
future trajectory from its recently observed motion. As of the research-scale
upgrade (Part 5), the default block is a *selective* S4D layer -- S4D's
diagonal linear SSM recurrence with a Mamba-style input-dependent
discretization step (see SelectiveS4DLayer below) -- at a genuinely
research-scale parameter count (~330k for ObstaclePredictor, ~210k for
MultimodalObstaclePredictor), not the ~27.6k-parameter demo-scale model this
project shipped with initially. The original plain-S4D blocks (S4DLayer,
S4DBlock) are kept in this file and remain selectable (`selective=False`) so
the two architectures can be compared directly rather than the smaller one
being silently deleted.

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


class SelectiveS4DLayer(nn.Module):
    """S4D with *selective* (Mamba-style) discretization: the effective
    per-step timestep dt is a learned function of the current input instead
    of a fixed constant.

    S4/S4D (above) discretizes its continuous-time SSM with a single dt fixed
    for every input, every timestep -- the model's effective "sampling rate"
    of its own internal dynamics never changes no matter what it's looking
    at. Mamba's central change (Gu & Dao, "Mamba: Linear-Time Sequence
    Modeling with Selective State Spaces," arXiv:2312.00752, 2023/2024) is to
    make the discretization step Delta a function of the input token itself,
    computed by a small learned projection: intuitively, the model learns to
    take a "bigger step" (let new input dominate, forget faster) on
    informative inputs and a "smaller step" (hold state, ignore) on
    uninformative ones -- an input-dependent, content-aware gate on the
    state recurrence, which is what "selective" refers to. This is the part
    of Mamba responsible for most of its improvement over plain S4 in the
    original paper's ablations.

    Full Mamba also makes the B and C projections input-dependent (a full
    "S6" scan); this implementation makes *only* Delta input-dependent, and
    keeps B/C as the same per-channel learned parameters as S4D. That's a
    deliberate scope reduction, not an oversight: this project already runs
    its SSM step-by-step (not through Mamba's parallel hardware-aware scan,
    which exists to make the recurrence trainable at GPU scale) since the
    sequences here are short (K=H=10) and this runs on CPU, so an
    input-dependent Delta is the one change that meaningfully matters here
    without a large increase in parameters/compute for a benefit this
    project's short sequences wouldn't exercise anyway.

    Delta is produced per-channel per-timestep as
    dt_base * sigmoid(Linear(u)), bounded to (0, 2*dt_base) and initialized
    (zero bias) to sigmoid(0) = 0.5 -> dt_base, i.e. identical to plain S4D's
    fixed dt at initialization, so training starts from the same behavior
    and *learns* whatever input-dependent deviation from it actually helps.
    """

    def __init__(self, d_model: int, d_state: int, dt: float):
        super().__init__()
        self.d_model, self.d_state, self.dt_base = d_model, d_state, dt
        self.log_decay = nn.Parameter(torch.rand(d_model, d_state) * 2 - 2)
        self.omega = nn.Parameter(torch.rand(d_model, d_state) * (math.pi / dt))
        self.B = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.zeros(d_model))
        # Selective discretization: dt_t = dt_base * sigmoid(delta_proj(u)).
        # Zero-init weight + bias => sigmoid(0) = 0.5 => dt_t = dt_base at
        # init, exactly matching plain S4D's fixed step (see docstring).
        self.delta_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def init_state(self, batch_size: int, device):
        a = torch.zeros(batch_size, self.d_model, self.d_state, device=device)
        b = torch.zeros(batch_size, self.d_model, self.d_state, device=device)
        return (a, b)

    def step(self, u: torch.Tensor, state):
        """u: (B, d_model) input at this timestep. Returns (y, new_state)."""
        a, b = state
        delta = self.dt_base * 2.0 * torch.sigmoid(self.delta_proj(u))   # (B, d_model)

        decay = F.softplus(self.log_decay) + 1e-3                        # (d_model, d_state)
        r = torch.exp(-decay.unsqueeze(0) * delta.unsqueeze(-1))          # (B, d_model, d_state)
        theta = self.omega.unsqueeze(0) * delta.unsqueeze(-1)             # (B, d_model, d_state)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        drive = delta.unsqueeze(-1) * self.B.unsqueeze(0) * u.unsqueeze(-1)  # (B, d_model, d_state)
        a_new = r * (cos_t * a - sin_t * b) + drive
        b_new = r * (sin_t * a + cos_t * b)

        y = (self.C_re * a_new - self.C_im * b_new).sum(-1) + self.D * u
        return y, (a_new, b_new)


class SelectiveS4DBlock(nn.Module):
    """Pre-norm selective-S4D layer + GLU + residual -- same block pattern as
    S4DBlock, with SelectiveS4DLayer swapped in for the SSM core."""

    def __init__(self, d_model: int, d_state: int, dt: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveS4DLayer(d_model, d_state, dt)
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
    def __init__(self, K: int = 10, H: int = 10, d_model: int = 160,
                 d_state: int = 40, n_layers: int = 3, dt: float = 0.1,
                 selective: bool = True):
        """selective=True (default, moderate research-scale ~330k params)
        uses SelectiveS4DBlock (Mamba-style input-dependent discretization,
        see its docstring); selective=False reproduces the original plain-
        S4D predictor (~27.6k params at the old default sizes) for anyone
        who wants to compare the two architectures directly."""
        super().__init__()
        self.K, self.H, self.dt = K, H, dt
        block_cls = SelectiveS4DBlock if selective else S4DBlock
        self.input_proj = nn.Linear(2, d_model)
        self.blocks = nn.ModuleList([block_cls(d_model, d_state, dt) for _ in range(n_layers)])
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


class MultimodalObstaclePredictor(nn.Module):
    """Same S4D encoder as ObstaclePredictor, but decodes M independent
    autoregressive rollouts ("modes") instead of one, plus a per-mode
    probability head. Built for genuinely ambiguous obstacles (see
    obstacle_trajectory_data.py's "branch" pattern) where a single correct
    answer doesn't exist during the ambiguous window: a unimodal model is
    mathematically stuck predicting something *between* the true outcomes,
    which matches neither of them.

    Unlike ObstaclePredictor, this decoder is trained fully autoregressively
    with NO teacher forcing at all: teacher-forcing one ground-truth future
    into M competing mode rollouts is ill-posed (which mode is "responsible"
    for matching it isn't known until after the fact -- see `mtp_loss`), so
    every mode always feeds its own last prediction back as the next input,
    during training exactly as at inference. That means there is no
    train/inference mismatch for this head to begin with -- no exposure
    bias, by construction, unlike the scheduled-sampling fix ObstaclePredictor
    needed (see its docstring) for exactly that problem.

    The M rollouts start from an *identical* encoded state (every mode has
    seen the same observed window -- there's nothing in the past to make one
    mode's encoding different from another's) and are differentiated only by
    a small learned per-mode input embedding added at every decode step:
    enough to let them diverge into different futures without giving each
    mode its own encoder, which would be extra capacity spent on a
    difference that only needs to show up in the decode.
    """

    def __init__(self, K: int = 10, H: int = 10, d_model: int = 128,
                 d_state: int = 32, n_layers: int = 3, dt: float = 0.1, n_modes: int = 2,
                 selective: bool = True):
        """selective=True (default, moderate research-scale ~210k params)
        uses SelectiveS4DBlock; selective=False reproduces the original
        plain-S4D multimodal predictor for direct comparison."""
        super().__init__()
        self.K, self.H, self.dt, self.n_modes = K, H, dt, n_modes
        block_cls = SelectiveS4DBlock if selective else S4DBlock
        self.input_proj = nn.Linear(2, d_model)
        self.blocks = nn.ModuleList([block_cls(d_model, d_state, dt) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, 2)
        # Init scale matters more than it looks like it should: too small
        # (0.1, tried first) and the M rollouts start out nearly identical,
        # so early "winner" assignment (see mtp_loss) is essentially noise
        # rather than a meaningful signal, and training tends to snowball
        # onto one dominant "generalist" mode that wins almost everywhere
        # instead of letting different modes specialize to different
        # outcomes -- a real instability hit while building this, not a
        # hypothetical one (see train_multimodal_predictor.py's notes and
        # the README's honest account of how well specialization actually
        # ended up working). 0.5 gives the modes enough of a head start to
        # differ meaningfully from the first few batches.
        self.mode_embed = nn.Parameter(torch.randn(n_modes, d_model) * 0.5)
        self.mode_head = nn.Linear(d_model, n_modes)

    def forward(self, past_disp: torch.Tensor):
        """past_disp: (B, K, 2) observed per-step displacements.
        Returns (pred_offsets, pred_disp, mode_logits):
          pred_offsets, pred_disp: (B, M, H, 2), one full rollout per mode.
          mode_logits: (B, M), unnormalized -- softmax for probabilities."""
        B = past_disp.shape[0]
        device = past_disp.device
        states = [blk.init_state(B, device) for blk in self.blocks]

        u = None
        for t in range(self.K):
            u = self.input_proj(past_disp[:, t, :])
            for i, blk in enumerate(self.blocks):
                u, states[i] = blk.step(u, states[i])
        ctx = u   # (B, d_model) -- pooled post-encoder context for the mode head
        # Detached deliberately: mode_head is trained (see train_multimodal_
        # predictor.py) against a target that is, for genuinely ambiguous
        # inputs, fundamentally unpredictable from ctx (that's the whole
        # point of "branch" -- see obstacle_trajectory_data.py). Letting
        # that loss backprop into the shared encoder was a real bug found
        # while building this: the encoder would get dragged around trying
        # to extract a signal that provably isn't there, which measurably
        # corrupted the (perfectly learnable) decoder specialization on top
        # of it. Stop-gradient here makes mode_head a pure readout that
        # cannot damage the representation the decoders depend on.
        mode_logits = self.mode_head(ctx.detach())

        last_obs_disp = past_disp[:, -1, :]
        all_offsets, all_disp = [], []
        for m in range(self.n_modes):
            mode_states = [(a.clone(), b.clone()) for a, b in states]
            prev_disp = last_obs_disp
            preds = []
            for h in range(self.H):
                u_in = self.input_proj(prev_disp) + self.mode_embed[m]
                for i, blk in enumerate(self.blocks):
                    u_in, mode_states[i] = blk.step(u_in, mode_states[i])
                pred_disp = self.output_proj(u_in)
                preds.append(pred_disp)
                prev_disp = pred_disp   # always autoregressive -- see class docstring
            mode_disp = torch.stack(preds, dim=1)               # (B, H, 2)
            mode_offsets = torch.cumsum(mode_disp, dim=1)
            all_disp.append(mode_disp)
            all_offsets.append(mode_offsets)

        pred_disp = torch.stack(all_disp, dim=1)         # (B, M, H, 2)
        pred_offsets = torch.stack(all_offsets, dim=1)   # (B, M, H, 2)
        return pred_offsets, pred_disp, mode_logits


def mtp_loss(pred_offsets: torch.Tensor, mode_logits: torch.Tensor,
             true_offsets: torch.Tensor, cls_weight: float = 1.0, epsilon: float = 0.0,
             forced_winner: torch.Tensor | None = None):
    """Standard "multiple-trajectory-prediction" winner-take-all loss (Cui et
    al., "Multimodal Trajectory Predictions for Autonomous Driving using Deep
    Convolutional Networks," ICRA 2019): for each sample, find the mode whose
    predicted trajectory is closest (MSE) to the true future -- the
    "responsible" mode -- and backpropagate regression loss through *only*
    that mode. The other modes get no gradient from this sample, which is
    what lets different modes specialize to different outcomes instead of
    all averaging toward the same answer (plain MSE across all modes would
    do exactly that averaging, and defeat the entire point of a mixture
    output). A cross-entropy term separately trains the mode-probability
    head to predict which mode will win, so `mode_logits` becomes a genuine
    (if imperfect) probability over outcomes rather than an unused output.

    Plain (epsilon=0, forced_winner=None) winner-take-all turned out to have
    a rich-get-richer failure mode while building this: a mode that wins
    slightly more often early (for essentially arbitrary, init-dependent
    reasons) gets all the regression gradient on those samples, gets better
    at them, and so wins even more -- which, on the "branch" pattern
    specifically, snowballed into both modes converging to nearly the same
    "generalist" output instead of specializing to the two true outcomes
    (confirmed by checking predicted final displacement per mode, not just
    which mode nominally "won" -- see the README for the numbers). Two
    things fixed it, both supported here:

      - `forced_winner` (B,): when an entry is >= 0, that value is used
        as the winner directly instead of arg-min -- for "branch" samples,
        train_multimodal_predictor.py passes the pattern's own known
        go/stop label (available at training time from the synthetic data
        generator, even though the model never sees it as input) as a
        privileged supervision signal. Pass -1 for samples that should
        still use ordinary arg-min competition (every non-ambiguous
        pattern -- there's only one true outcome, so it doesn't matter
        which mode "claims" it).
      - `epsilon`: with probability epsilon, the arg-min winner (for
        entries where forced_winner is -1 or absent) is replaced by a
        uniformly random mode, to keep gradient reaching under-used modes.
        Kept here for reuse/completeness; train_multimodal_predictor.py
        currently relies on forced_winner instead, since it is the more
        direct fix for the one pattern that actually needs specialization.

    pred_offsets: (B, M, H, 2). mode_logits: (B, M). true_offsets: (B, H, 2).
    Returns (loss, winner_idx, per_mode_mse) -- the latter two are useful for
    diagnosing mode collapse (all samples picking the same winner) during
    training."""
    err = pred_offsets - true_offsets.unsqueeze(1)              # (B, M, H, 2)
    per_mode_mse = (err ** 2).mean(dim=(2, 3))                  # (B, M)
    argmin_idx = per_mode_mse.argmin(dim=1)                     # (B,)
    if epsilon > 0:
        B, M = per_mode_mse.shape
        random_idx = torch.randint(0, M, (B,), device=per_mode_mse.device)
        use_random = torch.rand(B, device=per_mode_mse.device) < epsilon
        winner_idx = torch.where(use_random, random_idx, argmin_idx)
    else:
        winner_idx = argmin_idx
    if forced_winner is not None:
        winner_idx = torch.where(forced_winner >= 0, forced_winner, winner_idx)
    reg_loss = per_mode_mse.gather(1, winner_idx.unsqueeze(1)).mean()
    cls_loss = F.cross_entropy(mode_logits, winner_idx)
    return reg_loss + cls_weight * cls_loss, winner_idx, per_mode_mse
