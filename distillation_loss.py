"""
distillation_loss.py

Purpose:
--------
Implements the Branch-Level Knowledge Distillation loss for RepVGG training.

Components:
-----------
1. Output-Level Distillation (standard KD)
   - KL divergence between student and teacher final predictions

2. Branch-Level Distillation (proposed extension)
   - KL divergence between individual branch outputs and teacher predictions

3. Combined Loss
   - Weighted combination of CE loss, output-level KL, and branch-level KL
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training_utils import soft_target_cross_entropy


class BranchLevelDistillationLoss(nn.Module):
    """
    Branch-Level Knowledge Distillation Loss.

    Combines:
    1. Cross-Entropy Loss (classification objective)
    2. Output-Level Distillation (KL between final outputs)
    3. Branch-Level Distillation (KL between branch outputs and teacher)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha_ce: float = 1.0,
        alpha_output_kl: float = 2.0,
        alpha_branch_kl: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        temperature : float
            Temperature for softening distributions. Higher = softer.
        alpha_ce : float
            Weight for cross-entropy loss.
        alpha_output_kl : float
            Weight for output-level KL divergence.
        alpha_branch_kl : float
            Weight for branch-level KL divergence.
        label_smoothing : float
            Smoothing factor for the classification target when hard labels are used.
        """
        super().__init__()

        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_output_kl = alpha_output_kl
        self.alpha_branch_kl = alpha_branch_kl
        self.label_smoothing = label_smoothing

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.branch_projections = nn.ModuleDict()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        target_probs: torch.Tensor | None = None,
        student_branches: dict[str, dict] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined distillation loss.

        Parameters
        ----------
        student_logits : torch.Tensor
            Student model final output logits [batch_size, num_classes].
        teacher_logits : torch.Tensor
            Teacher model final output logits [batch_size, num_classes].
        labels : torch.Tensor
            Ground truth labels [batch_size].
        target_probs : torch.Tensor or None
            Soft targets [batch_size, num_classes], used when mixup is enabled.
        student_branches : dict or None
            Nested dict of student branch outputs. Structure:
            {
                "stage0": {"3x3": tensor, "1x1": tensor, ...},
                "stage1": {"block_0": {...}, "block_1": {...}, ...},
                ...
            }

        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all losses.
        loss_dict : dict
            Breakdown of individual loss components for logging.
        """
        loss_dict: dict[str, float] = {}

        # 1. Cross-Entropy Loss (standard classification)
        if target_probs is None:
            ce = self.ce_loss(student_logits, labels)
        else:
            ce = soft_target_cross_entropy(student_logits, target_probs)
        loss_dict["ce_loss"] = ce.item()

        # 2. Output-Level Distillation (final layer only)
        output_kl = self._compute_output_kl(student_logits, teacher_logits)
        loss_dict["output_kl"] = output_kl.item()

        # 3. Branch-Level Distillation (individual branches)
        branch_kl = torch.tensor(0.0, device=student_logits.device)
        if student_branches is not None:
            branch_kl = self._compute_branch_kl(student_branches, teacher_logits)
            loss_dict["branch_kl"] = branch_kl.item()

        # Combined loss
        total_loss = (
            self.alpha_ce * ce
            + self.alpha_output_kl * output_kl
            + self.alpha_branch_kl * branch_kl
        )

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict

    def initialize_branch_projections(
        self,
        student_branches: dict[str, dict],
        num_classes: int,
    ) -> None:
        """
        Create one reusable projection per exposed branch.

        Projections are part of the loss module so they can be optimized
        together with the student model instead of being recreated on each
        forward pass.
        """
        for branch_name, branch_output in self._iter_named_branch_outputs(
            student_branches
        ):
            projection_key = self._projection_key(branch_name)
            in_features = branch_output.shape[1]

            if projection_key in self.branch_projections:
                existing = self.branch_projections[projection_key]
                if isinstance(existing, nn.Linear) and existing.in_features != in_features:
                    raise ValueError(
                        f"Projection for branch {branch_name} expects "
                        f"{existing.in_features} features, got {in_features}."
                    )
                continue

            if in_features == num_classes:
                projection: nn.Module = nn.Identity()
            else:
                projection = nn.Linear(in_features, num_classes)

            self.branch_projections[projection_key] = projection.to(
                branch_output.device
            )

    def _compute_output_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence between student and teacher final outputs.
        """
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        kl = self.kl_loss(student_probs, teacher_probs)
        return kl

    def _compute_branch_kl(
        self,
        student_branches: dict[str, dict],
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence for each branch output against teacher.

        Branches are first averaged pooled to match final feature size,
        then passed through a projection to align with teacher output dim.
        """
        total_branch_kl = 0.0
        branch_count = 0

        device = teacher_logits.device

        for branch_name, branch_output in self._iter_named_branch_outputs(
            student_branches
        ):
            branch_kl = self._branch_output_to_kl(
                branch_name,
                branch_output,
                teacher_logits,
            )
            total_branch_kl = total_branch_kl + branch_kl
            branch_count += 1

        if branch_count == 0:
            return torch.tensor(0.0, device=device)

        return total_branch_kl / branch_count

    def _iter_named_branch_outputs(
        self,
        student_branches: dict[str, dict],
    ) -> list[tuple[str, torch.Tensor]]:
        named_branches: list[tuple[str, torch.Tensor]] = []

        for stage_name, stage_data in student_branches.items():
            if not isinstance(stage_data, dict):
                continue

            for key, value in stage_data.items():
                if isinstance(value, torch.Tensor):
                    named_branches.append((f"{stage_name}.{key}", value))
                elif isinstance(value, dict):
                    for block_branch_name, block_branch_output in value.items():
                        if isinstance(block_branch_output, torch.Tensor):
                            named_branches.append(
                                (
                                    f"{stage_name}.{key}.{block_branch_name}",
                                    block_branch_output,
                                )
                            )

        return named_branches

    def _projection_key(self, branch_name: str) -> str:
        return branch_name.replace(".", "__")

    def _branch_output_to_kl(
        self,
        branch_name: str,
        branch_output: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a branch feature map to KL loss against teacher.

        Assumes branch_output is [B, C, H, W].
        """
        # Global average pooling
        branch_pooled = F.adaptive_avg_pool2d(branch_output, output_size=1)
        branch_pooled = torch.flatten(branch_pooled, start_dim=1)

        projection_key = self._projection_key(branch_name)
        if projection_key not in self.branch_projections:
            raise RuntimeError(
                "Branch projections are not initialized. "
                "Call initialize_branch_projections(student_branches, num_classes) "
                "before using branch-level distillation."
            )

        projection = self.branch_projections[projection_key]
        branch_logits = projection(branch_pooled)

        # Compute KL divergence
        branch_probs = F.log_softmax(branch_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        kl = self.kl_loss(branch_probs, teacher_probs)
        return kl


def create_distillation_loss(
    temperature: float = 4.0,
    alpha_ce: float = 1.0,
    alpha_output_kl: float = 2.0,
    alpha_branch_kl: float = 1.0,
    label_smoothing: float = 0.0,
) -> BranchLevelDistillationLoss:
    """
    Factory function for branch-level distillation loss.
    """
    return BranchLevelDistillationLoss(
        temperature=temperature,
        alpha_ce=alpha_ce,
        alpha_output_kl=alpha_output_kl,
        alpha_branch_kl=alpha_branch_kl,
        label_smoothing=label_smoothing,
    )
