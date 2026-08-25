# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed tensor artifacts emitted alongside generated video."""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch import Tensor

_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
"""Portable artifact name accepted as a filename stem."""


@dataclass(frozen=True, slots=True)
class TensorArtifactSchema:
    """Describe one named tensor artifact produced by a session."""

    name: str
    """Stable name used to route and persist the artifact."""

    dimension_names: tuple[str, ...]
    """Semantic names for the tensor dimensions, in storage order."""

    concatenate_axis: int | None = 0
    """Axis joining outputs from multiple steps; ``None`` permits one output."""

    def __post_init__(self) -> None:
        if not _ARTIFACT_NAME.fullmatch(self.name):
            raise ValueError(
                "Tensor artifact names must start with an alphanumeric character "
                "and contain only alphanumerics, dots, underscores, or hyphens."
            )
        if any(not name.strip() for name in self.dimension_names):
            raise ValueError("Tensor artifact dimension names must be non-empty.")
        if len(set(self.dimension_names)) != len(self.dimension_names):
            raise ValueError("Tensor artifact dimension names must be unique.")
        if self.concatenate_axis is not None and not (
            -len(self.dimension_names)
            <= self.concatenate_axis
            < len(self.dimension_names)
        ):
            raise ValueError(
                "Tensor artifact concatenate_axis must identify a declared dimension."
            )


@dataclass(frozen=True, slots=True)
class TensorArtifactOutput:
    """Carry one tensor matching a declared artifact schema."""

    schema: TensorArtifactSchema
    """Schema identifying and describing the tensor."""

    tensor: Tensor
    """Artifact tensor. Sinks detach and transfer it before persistence."""

    def __post_init__(self) -> None:
        if self.tensor.ndim != len(self.schema.dimension_names):
            raise ValueError(
                f"Tensor artifact {self.schema.name!r} declares "
                f"{len(self.schema.dimension_names)} dimensions but received "
                f"shape {tuple(self.tensor.shape)}."
            )


__all__ = ["TensorArtifactOutput", "TensorArtifactSchema"]
