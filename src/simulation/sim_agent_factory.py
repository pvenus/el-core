# sim_engine/sim_agent_factory.py
from dataclasses import dataclass
from typing import Optional

from .sim_types import Vars, Vector, Meta
from .dto.vector_space import VectorSpaceSpec
from .dto.agent import AgentSpec, AgentState
from .sim_agent import SimAgent


@dataclass
class SimAgentFactory:
    default_comfort_radius: float = 1.0

    def create_from_space(
        self,
        space: VectorSpaceSpec,
        comfort_key: Optional[str] = None,
        comfort_radius: Optional[float] = None,
        spec_vars: Optional[Vars] = None,
        spec_meta: Optional[Meta] = None,
        init_vec: Optional[Vector] = None,
        init_vars: Optional[Vars] = None,
    ) -> SimAgent:
        space.validate()

        comfort_vec = [0.0] * space.dim
        if comfort_key is not None:
            hit = [a for a in space.axes if a.key == comfort_key]
            if not hit:
                raise ValueError(f"comfort_key '{comfort_key}' not found")
            if hit[0].proto_vec is not None:
                comfort_vec = list(hit[0].proto_vec)

        cr = float(comfort_radius) if comfort_radius is not None else float(self.default_comfort_radius)

        spec = AgentSpec(
            dim=space.dim,
            comfort_vec=comfort_vec,
            comfort_radius=cr,
            vars=dict(spec_vars or {}),
            meta={**dict(spec_meta or {}), "space_id": space.space_id},
        )

        state = AgentState(
            turn=0,
            current_vec=list(init_vec) if init_vec is not None else [0.0] * space.dim,
            vars=dict(init_vars) if init_vars is not None else dict(spec.vars),
        )

        return SimAgent(spec=spec, state=state)
