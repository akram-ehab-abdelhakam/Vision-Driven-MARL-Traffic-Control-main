"""
kl_adaptive_lr.py
=================
KL-Adaptive Learning Rate Controller for MAPPO-PIS-GAT.

Adjusts per-parameter-group learning rates based on the KL divergence
between the old policy (before the PPO update) and the new policy (after).

Default hyperparameters (matching requirements):
    actor_lr   = 3e-4   (policy head)
    critic_lr  = 1e-3   (centralized critic)
    gat_lr     = 1e-4   (GAT / intent-aggregator layers)
    kl_target  = 0.01   (desired per-update KL)
    lr_min     = 1e-5
    lr_max     = 5e-4

Adjustment rule (standard adaptive-KL):
    KL > 1.5 * kl_target  →  lr *= 0.7   (policy drifting too far)
    KL < 0.5 * kl_target  →  lr *= 1.3   (policy changing too little)
    otherwise              →  lr unchanged

Safeguards against runaway LR inflation:
    1. Degenerate guard:  KL ≤ 0.0 or NaN → skip (policy saturated or
       numerical issue).
    2. Deadband floor:    KL < kl_floor (default 1e-4) → "unchanged".
       Below this the estimator has no meaningful resolution.
    3. Consecutive cap:   After max_consecutive_increases (default 3)
       "increased" actions in a row without an intervening decrease/
       unchanged, refuse further increases.

Usage (inside MAPPOAgent / train_epoch)
---------------------------------------
See `build_kl_adaptive_optimizer()` for constructing the combined
actor+critic+GAT optimizer with three separate parameter groups, and
`KLAdaptiveLRController` for the per-step adjustment logic.

Where to insert in the training loop
--------------------------------------
After every call to optimizer.step() inside train_epoch, compute the
KL divergence between old_log_probs and the freshly-computed log_probs,
then call controller.step(kl). That is:

    1. Before optimizer.zero_grad() — stash old_log_probs (already done
       via old_lp_t in train_epoch).
    2. After loss.backward() + optimizer.step() — recompute log_probs
       with the updated parameters (or reuse the just-computed ones from
       the forward pass) and call controller.step(kl).

The controller modifies optimizer param_group['lr'] in-place, so the
next mini-batch immediately picks up the adjusted rate.

Implementation notes
---------------------
* KL is computed as the mean over the batch of:
      KL(old || new) = Σ_a  π_old(a|s) · [log π_old(a|s) − log π_new(a|s)]
  For a discrete policy (Categorical), this equals:
      mean( (exp(old_log_p) * (old_log_p - new_log_p)).sum(-1) )
  but since we only have the selected-action log-prob pairs we use the
  approximate scalar estimator:
      KL ≈ mean(old_log_p − new_log_p)
  This is the same estimator used by OpenAI's PPO implementation and
  is suitable for early-stopping / LR control.

* The GAT optimizer is kept separate from the actor/critic optimizer
  (as in the original code). `KLAdaptiveLRController` can wrap either
  one or both; pass `include_gat=True` to also scale the GAT lr.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
KL_TARGET_DEFAULT   = 0.01   # lowered from 0.02 — true_kl (full distributional) is typically smaller than the old approx_kl estimator, so a lower target keeps the controller from always firing "too low"
ACTOR_LR_DEFAULT    = 3e-4
CRITIC_LR_DEFAULT   = 1e-3
GAT_LR_DEFAULT      = 1e-4
LR_MIN              = 1e-5
LR_MAX              = 5e-4

# Adjustment multipliers
KL_HIGH_THRESHOLD   = 1.5   # KL > kl_target * this  →  reduce lr
KL_LOW_THRESHOLD    = 0.5   # KL < kl_target * this  →  increase lr
LR_DECREASE_FACTOR  = 0.7
LR_INCREASE_FACTOR  = 1.3

# Deadband floor: KL values below this are treated as "unchanged" rather
# than "too low."  At 1e-4, this is 1% of kl_target=0.01 — well below the
# KL_LOW_THRESHOLD band (0.005) but above floating-point noise and genuine
# saturation artifacts.  Rationale: for a discrete policy with ~4-8 actions,
# KL < 1e-4 means the per-step probability shift is < 0.01% — the
# estimator has no meaningful resolution here, and treating it as "too
# low" would trigger unbounded LR increases.
KL_FLOOR_DEFAULT    = 1e-4

# Safety cap on consecutive LR increases.  After this many "increased"
# actions in a row without a "decreased" or "unchanged" event, the
# controller refuses to increase further until the counter resets.
# With increase_factor=1.3,3 consecutive increases ≈ 2.2x the base LR,
# which for actor_lr=3e-4 already exceeds lr_max=5e-4 after clamping —
# so3 is effectively the ceiling before the controller becomes inert.
MAX_CONSEC_INCREASES_DEFAULT = 3


# ---------------------------------------------------------------------------
# KL estimator helpers
# ---------------------------------------------------------------------------

def compute_kl_divergence(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
) -> float:
    """Approximate per-step KL divergence from selected-action log-probs.

    Uses the scalar estimator:
        KL ≈ mean(old_lp − new_lp)

    This is non-negative in expectation and tracks how much the policy has
    moved.  It matches the PPO early-stopping convention used by OpenAI.

    Parameters
    ----------
    old_log_probs : Tensor, shape (batch,)
        Log-probabilities of selected actions under the *old* policy
        (collected during rollout, i.e. `old_lp_t`).
    new_log_probs : Tensor, shape (batch,)
        Log-probabilities of the *same* actions re-evaluated under the
        current (updated) policy parameters.

    Returns
    -------
    float
        Scalar KL estimate.
    """
    with torch.no_grad():
        kl = (old_log_probs - new_log_probs).mean()
        return float(kl.item())


def compute_kl_from_logits(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> float:
    """Full distributional KL(old || new) from action logits.

    KL(old || new) = Σ_a  π_old(a) · (log π_old(a) − log π_new(a))

    More accurate than the scalar estimator but requires the full logit
    vector.  Use this variant if you want to monitor true distributional
    drift without the high-variance single-sample approximation.

    Parameters
    ----------
    old_logits, new_logits : Tensor, shape (batch, action_dim)

    Returns
    -------
    float
        Mean per-sample KL divergence.
    """
    with torch.no_grad():
        old_log_p = torch.log_softmax(old_logits.float(), dim=-1)
        new_log_p = torch.log_softmax(new_logits.float(), dim=-1)
        old_p     = old_log_p.exp()
        kl_per_sample = (old_p * (old_log_p - new_log_p)).sum(dim=-1)
        return float(kl_per_sample.mean().item())


# ---------------------------------------------------------------------------
# KL-Adaptive LR Controller
# ---------------------------------------------------------------------------

class KLAdaptiveLRController:
    """Adjusts optimizer learning rates based on observed KL divergence.

    Wraps one or more `torch.optim.Optimizer` objects.  After each PPO
    mini-batch update call `controller.step(kl)` to let the controller
    check whether the KL is within the target band and scale LRs if not.

    Three safeguards prevent runaway LR inflation when the policy
    converges to a near-deterministic (saturated) distribution:

    1. Degenerate guard:  KL ≤ 0.0 or NaN → skip (policy has fully
       saturated or a numerical issue occurred).
    2. Deadband floor:    KL < kl_floor (default 1e-4) → "unchanged".
       Below this threshold the estimator has no meaningful resolution
       to distinguish genuine convergence from measurement noise.
    3. Consecutive cap:   After max_consecutive_increases (default 3)
       "increased" actions in a row without an intervening
       "decreased"/"unchanged", refuse further increases.

    Parameters
    ----------
    optimizers : list of (optimizer, group_indices) tuples
        Each entry is (opt, [group_idx, ...]).  Only the listed param
        groups of each optimizer are scaled.  Use group_idx=None to scale
        all groups.
    kl_target : float
        Desired KL per update step.
    lr_min, lr_max : float
        Absolute clamps on the learning rate after any adjustment.
    decrease_factor : float
        Multiplier when KL is too high (< 1.0).
    increase_factor : float
        Multiplier when KL is too low (> 1.0).
    kl_floor : float
        Deadband floor.  KL values below this are treated as "unchanged"
        rather than "too low."
    max_consecutive_increases : int
        Hard cap on how many times "increased" can fire in a row.
    verbose : bool
        Print a message whenever the LR is adjusted.
    """

    def __init__(
        self,
        optimizers: List[Tuple[optim.Optimizer, Optional[List[int]]]],
        kl_target:                  float = KL_TARGET_DEFAULT,
        lr_min:                     float = LR_MIN,
        lr_max:                     float = LR_MAX,
        decrease_factor:            float = LR_DECREASE_FACTOR,
        increase_factor:            float = LR_INCREASE_FACTOR,
        kl_floor:                   float = KL_FLOOR_DEFAULT,
        max_consecutive_increases:  int   = MAX_CONSEC_INCREASES_DEFAULT,
        verbose:                    bool  = False,
    ):
        self.optimizers      = optimizers
        self.kl_target       = kl_target
        self.lr_min          = lr_min
        self.lr_max          = lr_max
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor
        self.kl_floor                   = kl_floor
        self.max_consecutive_increases  = max_consecutive_increases
        self.verbose         = verbose

        # Diagnostics
        self.last_kl:     float = 0.0
        self.n_increased: int   = 0
        self.n_decreased: int   = 0
        self.n_unchanged: int   = 0
        self.n_degenerate: int  = 0   # batches where KL was ≤ 0, NaN, or below kl_floor

        # Consecutive-increase tracking (safety cap)
        self.consecutive_increases: int = 0

    # ------------------------------------------------------------------
    @property
    def is_lr_pinned(self) -> bool:
        """True if any controlled LR is currently at lr_max."""
        for opt, group_indices in self.optimizers:
            indices = group_indices if group_indices is not None \
                      else list(range(len(opt.param_groups)))
            for gi in indices:
                if opt.param_groups[gi]["lr"] >= self.lr_max:
                    return True
        return False

    # ------------------------------------------------------------------
    def step(self, kl: float, agent_id: str = "") -> str:
        """Check KL and update LRs if necessary.

        Parameters
        ----------
        kl : float
            KL divergence computed after the latest optimizer.step().
        agent_id : str, optional
            Label for verbose log messages (e.g. tls_id).

        Returns
        -------
        str
            One of "decreased", "increased", "unchanged".
        """
        self.last_kl = kl
        action = "unchanged"

        # ── Safeguard 1: degenerate KL (NaN, negative) ────────────────
        # A KL reading of exactly 0.0, negative, or NaN means the policy
        # logits have saturated or a numerical issue occurred.  Never
        # treat this as "too low."
        if kl <= 0.0 or math.isnan(kl):
            self.n_degenerate += 1
            if self.verbose:
                label = f"[{agent_id}] " if agent_id else ""
                print(
                    f"  [KL-LR] {label}DEGENERATE kl={kl} -- "
                    f"skipping LR adjust (n_degenerate={self.n_degenerate})",
                    flush=True,
                )
            return "unchanged"

        # ── Safeguard 2: deadband floor ───────────────────────────────
        # Below kl_floor the estimator has no meaningful resolution to
        # distinguish genuine convergence from floating-point noise.
        # Treat as "unchanged" so we don't trigger LR increases on
        # degenerate-magnitude signals.
        if kl < self.kl_floor:
            self.n_degenerate += 1
            self.consecutive_increases = 0   # reset cap on any non-increase
            if self.verbose:
                label = f"[{agent_id}] " if agent_id else ""
                print(
                    f"  [KL-LR] {label}BELOW_FLOOR kl={kl:.2e} "
                    f"< floor={self.kl_floor:.2e} -- "
                    f"treating as unchanged (n_degenerate={self.n_degenerate})",
                    flush=True,
                )
            return "unchanged"

        # ── Normal KL band logic ──────────────────────────────────────
        if kl > KL_HIGH_THRESHOLD * self.kl_target:
            factor = self.decrease_factor
            action = "decreased"
            self.n_decreased += 1
            self.consecutive_increases = 0   # reset cap on decrease
        elif kl < KL_LOW_THRESHOLD * self.kl_target:
            # ── Safeguard 3: consecutive-increase cap ─────────────────
            if self.consecutive_increases >= self.max_consecutive_increases:
                # Cap reached — refuse further increases until a
                # "decreased" or "unchanged" resets the counter.
                self.n_unchanged += 1
                if self.verbose:
                    label = f"[{agent_id}] " if agent_id else ""
                    print(
                        f"  [KL-LR] {label}CONSEC_CAP kl={kl:.5f} "
                        f"consecutive_increases={self.consecutive_increases} "
                        f">= cap={self.max_consecutive_increases} -- "
                        f"refusing further LR increase",
                        flush=True,
                    )
                return "unchanged"
            factor = self.increase_factor
            action = "increased"
            self.n_increased += 1
            self.consecutive_increases += 1
        else:
            self.n_unchanged += 1
            self.consecutive_increases = 0   # reset cap in the normal band
            return action

        # Apply the factor to every targeted param group
        for opt, group_indices in self.optimizers:
            indices = group_indices if group_indices is not None \
                      else list(range(len(opt.param_groups)))
            for gi in indices:
                old_lr = opt.param_groups[gi]["lr"]
                new_lr = float(
                    max(self.lr_min,
                        min(self.lr_max, old_lr * factor))
                )
                opt.param_groups[gi]["lr"] = new_lr
                if self.verbose and new_lr != old_lr:
                    label = f"[{agent_id}] " if agent_id else ""
                    print(
                        f"  [KL-LR] {label}group={gi}  "
                        f"kl={kl:.5f}  kl_target={self.kl_target:.4f}  "
                        f"action={action}  "
                        f"consec_inc={self.consecutive_increases}  "
                        f"lr: {old_lr:.2e} → {new_lr:.2e}",
                        flush=True,
                    )
        return action

    # ------------------------------------------------------------------
    def get_lrs(self) -> dict:
        """Return a snapshot of current LRs and controller diagnostics."""
        snapshot = {}
        for opt_idx, (opt, group_indices) in enumerate(self.optimizers):
            indices = group_indices if group_indices is not None \
                      else list(range(len(opt.param_groups)))
            for gi in indices:
                snapshot[f"opt{opt_idx}_group{gi}"] = opt.param_groups[gi]["lr"]
        snapshot["consecutive_increases"] = self.consecutive_increases
        snapshot["is_lr_pinned"] = self.is_lr_pinned
        return snapshot

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialisable state for checkpointing."""
        return {
            "last_kl":                  self.last_kl,
            "n_increased":              self.n_increased,
            "n_decreased":              self.n_decreased,
            "n_unchanged":              self.n_unchanged,
            "n_degenerate":             self.n_degenerate,
            "consecutive_increases":    self.consecutive_increases,
            "kl_target":                self.kl_target,
            "kl_floor":                 self.kl_floor,
            "lr_min":                   self.lr_min,
            "lr_max":                   self.lr_max,
            "max_consecutive_increases": self.max_consecutive_increases,
            # LR snapshot at save time
            "lrs": self.get_lrs(),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore counters (does NOT restore optimizer LRs — load those separately)."""
        self.last_kl                   = sd.get("last_kl",                   0.0)
        self.n_increased               = sd.get("n_increased",               0)
        self.n_decreased               = sd.get("n_decreased",               0)
        self.n_unchanged               = sd.get("n_unchanged",               0)
        self.n_degenerate              = sd.get("n_degenerate",              0)
        self.consecutive_increases     = sd.get("consecutive_increases",     0)
        # Allow checkpoint overrides for tunable thresholds
        self.kl_floor                  = sd.get("kl_floor", self.kl_floor)
        self.max_consecutive_increases = sd.get(
            "max_consecutive_increases", self.max_consecutive_increases)


# ---------------------------------------------------------------------------
# Convenience factory: build optimizer with three parameter groups
# ---------------------------------------------------------------------------

def build_kl_adaptive_optimizer(
    actor:          nn.Module,
    critic:         nn.Module,
    gat_aggregator: nn.Module,
    actor_lr:  float = ACTOR_LR_DEFAULT,
    critic_lr: float = CRITIC_LR_DEFAULT,
    gat_lr:    float = GAT_LR_DEFAULT,
) -> Tuple[optim.Adam, optim.Adam, "KLAdaptiveLRController"]:
    """Build separate actor/critic and GAT optimizers with KL-adaptive control.

    Returns three objects:
        actor_critic_opt  — Adam optimizer with two param groups:
                            group 0 = policy head (actor_lr)
                            group 1 = critic      (critic_lr)
        gat_opt           — Adam optimizer for GAT (gat_lr, group 0)
        kl_controller     — KLAdaptiveLRController wrapping both optimizers

    The controller scales all three groups independently when called.

    Notes
    -----
    The original codebase keeps actor_optimizer and critic_optimizer
    separate, and gat_optimizer is a third independent optimizer.
    This factory keeps that topology but adds the KL controller on top.
    If you prefer a single unified optimizer see the inline note below.
    """
    # -- Actor: feature net + action head + intent head ------------------
    actor_params = list(actor.parameters())

    # -- Critic parameters -----------------------------------------------
    critic_params = list(critic.parameters())

    # -- GAT aggregator --------------------------------------------------
    gat_params = list(gat_aggregator.parameters())
    if not gat_params:
        # Non-parametric aggregator (Mean / NoSharing) — use a dummy param
        # so the optimizer is still valid.
        _dummy = torch.zeros(1, requires_grad=True)
        gat_params = [_dummy]

    # Two separate optimizers to match the original code structure
    actor_opt  = optim.Adam(actor_params,  lr=actor_lr)
    critic_opt = optim.Adam(critic_params, lr=critic_lr)
    gat_opt    = optim.Adam(gat_params,    lr=gat_lr)

    # KL controller wraps all three optimizers; each has a single group (index 0)
    kl_ctrl = KLAdaptiveLRController(
        optimizers=[
            (actor_opt,  [0]),   # group 0 → actor lr
            (critic_opt, [0]),   # group 0 → critic lr
            (gat_opt,    [0]),   # group 0 → GAT lr
        ],
        kl_target       = KL_TARGET_DEFAULT,
        lr_min          = LR_MIN,
        lr_max          = LR_MAX,
        decrease_factor = LR_DECREASE_FACTOR,
        increase_factor = LR_INCREASE_FACTOR,
        verbose         = True,
    )

    return actor_opt, critic_opt, gat_opt, kl_ctrl


# ---------------------------------------------------------------------------
# KL-Adaptive Clip-Ratio Controller
# ---------------------------------------------------------------------------

class KLAdaptiveClipController:
    """Adjusts clip ratio based on observed KL divergence.

    Same adjustment rule as KLAdaptiveLRController but modifies clip_ratio
    instead of learning rates.  The clip_ratio is stored in a mutable
    *container* (a list with one element) so the controller can update it
    in-place and parent code reads it transparently.

    KL > kl_target * high_mult  →  clip *= clip_down   (tighten clipping)
    KL < kl_target * low_mult   →  clip *= clip_up     (loosen clipping)
    otherwise                    →  clip unchanged

    Parameters
    ----------
    clip_holder : list[float]
        A single-element list holding the current clip ratio.
        The controller writes to ``clip_holder[0]``.
    kl_target : float
        Desired KL per update step.
    clip_min, clip_max : float
        Absolute bounds on the clip ratio.
    high_mult : float
        KL > kl_target * high_mult triggers a decrease.
    low_mult : float
        KL < kl_target * low_mult triggers an increase.
    clip_down : float
        Multiplier when KL is too high (< 1.0).
    clip_up : float
        Multiplier when KL is too low (> 1.0).
    verbose : bool
        Print a message whenever the clip ratio is adjusted.
    """

    def __init__(
        self,
        clip_holder: list,
        kl_target:  float = KL_TARGET_DEFAULT,
        clip_min:   float = 0.05,
        clip_max:   float = 0.40,
        high_mult:  float = KL_HIGH_THRESHOLD,
        low_mult:   float = KL_LOW_THRESHOLD,
        clip_down:  float = 0.80,
        clip_up:    float = 1.20,
        verbose:    bool  = False,
    ):
        self.clip_holder = clip_holder
        self.kl_target   = kl_target
        self.clip_min    = clip_min
        self.clip_max    = clip_max
        self.high_mult   = high_mult
        self.low_mult    = low_mult
        self.clip_down   = clip_down
        self.clip_up     = clip_up
        self.verbose     = verbose

        # Diagnostics
        self.last_kl:     float = 0.0
        self.n_increased: int   = 0
        self.n_decreased: int   = 0
        self.n_unchanged: int   = 0

    # ------------------------------------------------------------------
    def step(self, kl: float, agent_id: str = "") -> str:
        """Check KL and adjust clip ratio if necessary.

        Parameters
        ----------
        kl : float
            KL divergence computed after the latest PPO update.
        agent_id : str, optional
            Label for verbose log messages (e.g. tls_id).

        Returns
        -------
        str
            One of ``"decreased"``, ``"increased"``, ``"unchanged"``.
        """
        self.last_kl = kl
        action = "unchanged"

        if kl > self.high_mult * self.kl_target:
            factor = self.clip_down
            action = "decreased"
            self.n_decreased += 1
        elif kl < self.low_mult * self.kl_target:
            factor = self.clip_up
            action = "increased"
            self.n_increased += 1
        else:
            self.n_unchanged += 1
            return action

        old_clip = self.clip_holder[0]
        new_clip = max(self.clip_min, min(self.clip_max, old_clip * factor))
        self.clip_holder[0] = new_clip

        if self.verbose and new_clip != old_clip:
            label = f"[{agent_id}] " if agent_id else ""
            print(
                f"  [KL-CLIP] {label}"
                f"kl={kl:.5f}  kl_target={self.kl_target:.4f}  "
                f"clip: {old_clip:.4f} -> {new_clip:.4f}  "
                f"action={action}",
                flush=True,
            )
        return action

    # ------------------------------------------------------------------
    def get_clip(self) -> float:
        """Return the current clip ratio from the mutable holder."""
        return self.clip_holder[0]

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialisable state for checkpointing."""
        return {
            "last_kl":     self.last_kl,
            "n_increased": self.n_increased,
            "n_decreased": self.n_decreased,
            "n_unchanged": self.n_unchanged,
            "clip":        self.clip_holder[0],
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore counters and clip ratio."""
        self.last_kl     = sd.get("last_kl",     0.0)
        self.n_increased = sd.get("n_increased",  0)
        self.n_decreased = sd.get("n_decreased",  0)
        self.n_unchanged = sd.get("n_unchanged",  0)
        self.clip_holder[0] = sd.get("clip", self.clip_holder[0])
