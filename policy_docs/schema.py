"""Structured output for R&D tax-incentive reform extraction."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Instrument = Literal[
    "volume_credit", "incremental_credit", "enhanced_deduction",
    "payroll_withholding_relief", "patent_box", "accelerated_depreciation",
    "other"]
Direction = Literal["introduction", "expansion", "restriction", "abolition"]
FirmScope = Literal["all", "sme_only", "large_only", "young_innovative"]


class ReformEvent(BaseModel):
    year_effective: int = Field(description="Year the change took effect")
    year_announced: int | None = Field(description="Year it was legislated or announced, if stated")
    instrument: Instrument
    direction: Direction
    firm_scope: FirmScope
    rate_before: float | None = Field(description="Rate before (e.g. 1.5 for a 150 % deduction, 0.2 for a 20 % credit), if stated")
    rate_after: float | None = Field(description="Rate after, if stated")
    summary: str = Field(description="One sentence: what changed")
    quote: str = Field(description="Verbatim sentence from the document that supports the event")
    page: int = Field(description="1-based page of the quote in the document")
    confidence: Literal["low", "medium", "high"]


class DocumentExtraction(BaseModel):
    country: str
    source_title: str
    events: list[ReformEvent]
