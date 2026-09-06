#!/usr/bin/env python3
"""WeMM-Embedding 어댑터 — 벤치마크 전용.

tools/bench_vv_quality.py 가 기대하는 SigLIP2Encoder 인터페이스
(encode_image / encode_text / model_name / dimensions / unload)를
tencent/WeMM-Embedding-* 위에 얇게 씌운다.

프로덕션 경로(backend/vector/)는 건드리지 않는다. 이 파일의 목적은
현행 pro 티어 인코더(siglip2-so400m-patch14-384, 84.7/100)와 WeMM 을
**같은 하네스·같은 표본**으로 비교하는 것 하나다.

WeMM 이 SigLIP2 와 다른 점:
  - 이미지와 텍스트가 하나의 공간에 들어간다(SigLIP2 도 그렇지만 WeMM 은
    비디오·문서·혼합 입력까지 같은 공간에 넣는다).
  - 반환 임베딩이 이미 L2 정규화돼 있다(모델 카드 명시). 그래도 하네스가
    코사인 유사도를 전제하므로 방어적으로 한 번 더 정규화한다.
  - 2B=2048, 9B=4096 차원. SigLIP2 pro 티어는 1152 차원이라 차원 수가
    다르다 — 점수는 비교 가능하지만 벡터를 섞을 수는 없다.
"""

from __future__ import annotations

import gc
from typing import Optional

import numpy as np
from PIL import Image

DEFAULT_MODEL = "tencent/WeMM-Embedding-2B"


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class WeMMEncoder:
    """SigLIP2Encoder 와 동일한 표면을 가진 WeMM 래퍼."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_MODEL
        self._model = None
        self._dim: Optional[int] = None

    # ── lifecycle ──────────────────────────────────────────────
    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        # trust_remote_code: WeMM 은 modeling_st_wemm.py 를 함께 배포한다.
        self._model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def unload(self):
        self._model = None
        self._dim = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def dimensions(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode_text("dimension probe").shape[0])
        return self._dim

    # ── encoding ───────────────────────────────────────────────
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """PIL 이미지 → L2 정규화 임베딩 (dim,)."""
        self._load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        vec = self._model.encode([image])
        return _l2(np.asarray(vec)[0])

    def encode_text(self, text: str) -> np.ndarray:
        """텍스트 → L2 정규화 임베딩 (dim,). 이미지와 같은 공간."""
        self._load()
        vec = self._model.encode([text])
        return _l2(np.asarray(vec)[0])


def build_encoder(model_name: Optional[str]):
    """모델 이름을 보고 적절한 인코더를 만든다.

    bench_vv_quality.py 가 `--encoder-model` 로 WeMM 을 받으면 이쪽,
    아니면 기존 SigLIP2Encoder 로 보낸다.
    """
    name = model_name or ""
    if "wemm" in name.lower():
        return WeMMEncoder(model_name)
    from backend.vector.siglip2_encoder import SigLIP2Encoder
    return SigLIP2Encoder(model_name=model_name)
