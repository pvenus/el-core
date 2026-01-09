from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Literal

from .dto.vector_space import VectorSpaceSpec


SourceType = Literal["memory", "text", "json"]


@dataclass
class SimSpaceProvider:
    """
    단일 Provider.
    - 인스턴스는 하나의 source_type만 가진다.
    - from_* 생성기로 인스턴스 교체만 하면 로직 교체 가능.
    """
    source_type: SourceType
    space_id: str = "ui_demo"

    # source payload (모드별로 하나만 사용)
    _memory_spec: Optional[VectorSpaceSpec] = None
    _text: Optional[str] = None
    _json_path: Optional[str] = None

    # ---------- constructors ----------
    @classmethod
    def from_memory(cls, spec: Optional[VectorSpaceSpec] = None) -> "SimSpaceProvider":
        return cls(source_type="memory", _memory_spec=spec)

    @classmethod
    def from_text(cls, text: str, space_id: str = "ui_demo") -> "SimSpaceProvider":
        return cls(source_type="text", space_id=space_id, _text=text)

    @classmethod
    def from_json(cls, path: str) -> "SimSpaceProvider":
        return cls(source_type="json", _json_path=path)

    # ---------- behavior ----------
    def load(self) -> VectorSpaceSpec:
        if self.source_type == "memory":
            if self._memory_spec is None:
                raise ValueError("SimSpaceProvider(memory): spec is None")
            return self._memory_spec

        if self.source_type == "text":
            if not self._text:
                raise ValueError("SimSpaceProvider(text): text is empty")
            return VectorSpaceSpec.from_text_lines(self._text, space_id=self.space_id)

        if self.source_type == "json":
            if not self._json_path:
                raise ValueError("SimSpaceProvider(json): path is empty")
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return VectorSpaceSpec.from_dict(data)

        raise RuntimeError(f"Unknown source_type: {self.source_type}")

    def save(self, spec: VectorSpaceSpec) -> None:
        spec.validate()

        if self.source_type == "memory":
            self._memory_spec = spec
            return

        if self.source_type == "text":
            # text provider는 저장 대상이 아니라는 철학이면 에러
            raise NotImplementedError("SimSpaceProvider(text) does not support save()")

        if self.source_type == "json":
            if not self._json_path:
                raise ValueError("SimSpaceProvider(json): path is empty")
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
            return

        raise RuntimeError(f"Unknown source_type: {self.source_type}")
