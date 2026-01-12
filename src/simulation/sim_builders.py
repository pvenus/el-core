from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.simulation.dto.vector_space import AxisSpec, VectorSpaceSpec
from src.simulation.dto.agent import AgentSpec
from src.simulation.sim_protocols import AgentSpecSource


# =========================
# JSON I/O (save/load)
# =========================

def save_space_spec_json(space: VectorSpaceSpec, path: str | Path, *, indent: int = 2) -> None:
    p = Path(path)
    p.write_text(json.dumps(space.to_dict(), ensure_ascii=False, indent=indent), encoding="utf-8")


def load_space_spec_json(path: str | Path) -> VectorSpaceSpec:
    p = Path(path)
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return VectorSpaceSpec.from_dict(data)


def save_agent_specs_json(agent_specs: list[AgentSpec], space_dim: int, path: str | Path, *, indent: int = 2) -> None:
    p = Path(path)
    payload = {
        "type": "AgentSpecList",
        "version": 1,
        "dim": int(space_dim),
        "agents": [a.to_dict() for a in agent_specs],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")


def load_agent_specs_json(path: str | Path, *, expected_dim: int | None = None) -> list[AgentSpec]:
    p = Path(path)
    payload: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

    dim = int(payload.get("dim") or 0)
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"AgentSpecList dim mismatch. expected={expected_dim}, got={dim}")

    agents_raw = payload.get("agents")
    if not isinstance(agents_raw, list):
        raise ValueError("AgentSpecList requires 'agents' list")

    return [AgentSpec.from_dict(a, dim=dim) for a in agents_raw]


# =========================
# Demo Builders (NO RANDOM)
# =========================

@dataclass
class DemoSpaceBuilder:
    """
    랜덤 제거: 하드코딩된 proto_vec 제공
    - dim은 고정 7 (필요하면 여기만 바꾸면 됨)
    - proto_vec은 one-hot basis (축 의미는 나중에 교체 가능)
    """
    dim: int = 7

    def build(self) -> VectorSpaceSpec:
        dim = self.dim
        axes: list[AxisSpec] = []
        for i in range(dim):
            v = np.zeros((dim,), dtype=float)
            v[i] = 1.0  # one-hot (고정)
            axes.append(
                AxisSpec(
                    key=f"axis_{i:02d}",
                    tags=[],
                    proto_vec=v,
                )
            )
        return VectorSpaceSpec(dim=dim, axes=axes)


@dataclass
class DemoAgentSpecSource(AgentSpecSource):
    """
    랜덤 제거: 에이전트 1개 하드코딩(리스트는 유지)
    """
    agent_id: str = "agent_01"
    name: str = "Agent 01"
    radius: float = 1.5

    def build_agent_specs(self, space: VectorSpaceSpec) -> list[AgentSpec]:
        # comfort_vec: 고정(전부 0)
        comfort = np.zeros((space.dim,), dtype=float)
        return [
            AgentSpec(
                agent_id=self.agent_id,
                name=self.name,
                comfort_vec=comfort,
                radius=float(self.radius),
            )
        ]


def build_demo_step_input_json(*, space_dim: int, agent_id: str) -> dict[str, Any]:
    """
    랜덤 제거: 하드코딩 StepInput JSON 생성
    - UI에서 JSON 입력 예시로도 사용 가능
    """
    direction = [0.0] * space_dim
    # 예: axis_00 방향으로 이동
    if space_dim > 0:
        direction[0] = 1.0

    return {
        "metadata": {"src": "demo_json"},
        "impacts_by_agent": {
            agent_id: [
                {
                    "direction": direction,
                    "magnitude": 1.0,
                    "duration": 1,
                    # delta_vars/profile는 보류(미사용) -> 생략 가능
                }
            ]
        },
    }


# (UI에서 specs를 그대로 source로 쓰고 싶을 때)
@dataclass
class ListAgentSpecSource(AgentSpecSource):
    specs: list[AgentSpec]

    def build_agent_specs(self, space: VectorSpaceSpec) -> list[AgentSpec]:
        for s in self.specs:
            if s.comfort_vec.shape != (space.dim,):
                raise ValueError("AgentSpec dim mismatch with VectorSpaceSpec")
        return list(self.specs)
