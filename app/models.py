from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class OverlayConfig(BaseModel):
    asset_url: str
    x: int
    y: int
    start_time: float  # seconds
    duration: float    # seconds


class ProgressBarConfig(BaseModel):
    color: str = "#FFFFFF"
    height_px: int = 4


class DynamicTextConfig(BaseModel):
    word: str
    start_time: float
    duration: float = 0.5


class JobConfig(BaseModel):
    transcription_provider: Literal["whisper", "assemblyai"] = "whisper"
    face_tracking: bool = False
    noise_reduction: bool = False
    background_music_url: Optional[str] = None
    caption_style: Literal["default", "highlight"] = "default"
    overlays: List[OverlayConfig] = []
    progress_bar: ProgressBarConfig = Field(default_factory=ProgressBarConfig)
    dynamic_text_words: List[DynamicTextConfig] = []


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: float = Field(ge=0.0, le=100.0)  # 0.0 – 100.0
    current_stage: Optional[str] = None
    output_urls: Optional[List[str]] = None   # populated when status == "completed"
    error: Optional[str] = None               # populated when status == "failed"


class WordToken(BaseModel):
    word: str
    start: float   # seconds
    end: float     # seconds
    confidence: float


class Transcript(BaseModel):
    job_id: str
    words: List[WordToken]
    is_empty: bool


class Segment(BaseModel):
    segment_id: str
    job_id: str
    start: float              # seconds
    end: float                # seconds
    score: float              # 0.0 – 1.0 composite score
    llm_virality_score: float
    rank: int
