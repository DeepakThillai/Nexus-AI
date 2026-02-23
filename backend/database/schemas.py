"""
schemas.py — Pydantic v2 models that mirror the MongoDB schema exactly.
All request/response types for FastAPI endpoints are defined here.
No business logic — pure data contracts.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field


# ═══════════════════════════════════════════════════════════════════
#  SUB-SCHEMAS (nested inside main user document)
# ═══════════════════════════════════════════════════════════════════

class ProfileSchema(BaseModel):
    target_role: Optional[str] = None
    skills: List[str] = []
    strengths: List[str] = []
    weaknesses: List[str] = []
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    experience_years: int = 0
    resume_uploaded: bool = False
    resume_uploaded_at: Optional[str] = None
    resume_file_name: Optional[str] = None


class CareerStateSchema(BaseModel):
    current_target_role: Optional[str] = None
    role_history: List[str] = []


class ReadinessAssessmentSchema(BaseModel):
    score: int = 0
    status: str = ""
    evaluation_summary: str = ""
    safer_adjacent_roles: List[str] = []
    advanced_adjacent_roles: List[str] = []


class RoadmapActionSchema(BaseModel):
    action_id: str
    week: int
    action_title: str
    status: str = "pending"
    score: Optional[int] = None


class RoadmapStepSchema(BaseModel):
    month: int
    step_title: str
    actions: List[RoadmapActionSchema] = []


class ActiveRoadmapSchema(BaseModel):
    generated_for_role: str = ""
    steps: List[RoadmapStepSchema] = []
    status: str = "not_started"


class ProgressSchema(BaseModel):
    actions_completed: int = 0
    actions_failed: int = 0
    weeks_completed: int = 0
    last_activity_at: Optional[str] = None


class RerouteStateSchema(BaseModel):
    is_active: bool = False
    reroute_count: int = 0
    original_roadmap_id: Optional[str] = None
    reroute_reason: Optional[str] = None
    reroute_options: List[str] = []
    selected_option: Optional[str] = None
    rerouted_at: Optional[str] = None
    can_return_to_previous: bool = True


class CurrentActionsSchema(BaseModel):
    this_week: List[Any] = []
    upcoming: List[Any] = []
    blocked: List[Any] = []


class ResumeAnalysisSchema(BaseModel):
    parsed_profile: Dict[str, Any] = {}
    extracted_skills: Dict[str, Any] = {}
    normalized_skills: List[str] = []
    skill_gap_analysis: Dict[str, Any] = {}
    career_recommendations: Dict[str, Any] = {}
    system_confidence: float = 0.0
    processing_time_seconds: float = 0.0


class MetadataSchema(BaseModel):
    total_sessions: int = 0
    agent_interaction_count: Dict[str, Any] = {}
    system_events: List[Any] = []


# ═══════════════════════════════════════════════════════════════════
#  MAIN USER DOCUMENT
# ═══════════════════════════════════════════════════════════════════

class UserDocument(BaseModel):
    """Full MongoDB user document — matches schema exactly."""
    user_id: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    profile: ProfileSchema = Field(default_factory=ProfileSchema)
    career_state: CareerStateSchema = Field(default_factory=CareerStateSchema)
    readiness_assessment: ReadinessAssessmentSchema = Field(default_factory=ReadinessAssessmentSchema)
    active_roadmap: ActiveRoadmapSchema = Field(default_factory=ActiveRoadmapSchema)
    progress: ProgressSchema = Field(default_factory=ProgressSchema)
    reroute_state: RerouteStateSchema = Field(default_factory=RerouteStateSchema)
    current_actions: CurrentActionsSchema = Field(default_factory=CurrentActionsSchema)
    resume_analysis: ResumeAnalysisSchema = Field(default_factory=ResumeAnalysisSchema)
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

    # Internal confidence (maps to CLI confidence_score)
    confidence_score: int = 0


# ═══════════════════════════════════════════════════════════════════
#  REQUEST SCHEMAS (API inputs)
# ═══════════════════════════════════════════════════════════════════

class OnboardRequest(BaseModel):
    """POST /api/onboard"""
    name: str
    email: str
    target_role: str
    skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    experience_years: int = 0
    phone: Optional[str] = None


class ReadinessStartRequest(BaseModel):
    """POST /api/readiness/start — generates 10 questions"""
    user_id: str


class ReadinessEvaluateRequest(BaseModel):
    """POST /api/readiness/evaluate — submits 10 answers"""
    user_id: str
    answers: List[str]  # ordered list of 10 answers


class ActionAssessRequest(BaseModel):
    """POST /api/action/assess"""
    user_id: str
    action_id: str
    answers: List[str]  # 10 answers to action questions


class ActionQuestionsRequest(BaseModel):
    """POST /api/action/questions — get questions for an action"""
    user_id: str
    action_id: str


class RerouteRequest(BaseModel):
    """POST /api/reroute"""
    user_id: str
    new_role: Optional[str] = None   # if user picks a role; null = just analyse


class FeedbackRequest(BaseModel):
    """POST /api/feedback"""
    user_id: str


class RoadmapRegenerateRequest(BaseModel):
    """POST /api/roadmap/regenerate"""
    user_id: str
    target_role: Optional[str] = None  # override role if switching


class HandsOnChatRequest(BaseModel):
    """POST /api/hands-on/chat"""
    user_id: str
    message: str
    conversation_history: List[Dict[str, str]] = []  # full history from frontend


# ═══════════════════════════════════════════════════════════════════
#  RESPONSE SCHEMAS (API outputs)
# ═══════════════════════════════════════════════════════════════════

class OnboardResponse(BaseModel):
    user_id: str
    message: str
    profile: ProfileSchema
    exists: bool  # REQUIRED - must always be present!


class ReadinessQuestionsResponse(BaseModel):
    user_id: str
    questions: List[str]
    # Stored server-side for evaluation (not sent again)


class ReadinessResultResponse(BaseModel):
    user_id: str
    score: int
    status: str
    evaluation_summary: str
    safer_adjacent_roles: List[str]
    advanced_adjacent_roles: List[str]
    confidence_score: int


class DashboardResponse(BaseModel):
    user_id: str
    profile: ProfileSchema
    confidence_score: int
    readiness: ReadinessAssessmentSchema
    progress: ProgressSchema
    roadmap_summary: Dict[str, Any]
    reroute_state: RerouteStateSchema
    career_state: CareerStateSchema
    weekly_scores: List[Dict[str, Any]]   # for line chart


class RoadmapResponse(BaseModel):
    user_id: str
    roadmap: ActiveRoadmapSchema
    confidence_score: int


class ActionQuestionsResponse(BaseModel):
    action_id: str
    action_title: str
    questions: List[str]


class ActionAssessResponse(BaseModel):
    action_id: str
    score: int
    passed: bool
    evaluation_summary: str
    updated_confidence_score: int
    reroute_check: Dict[str, Any]


class MarketResponse(BaseModel):
    user_id: str
    market_analysis: Dict[str, Any]


class RerouteResponse(BaseModel):
    user_id: str
    reroute_suggestion: bool
    suggested_roles: List[str]
    return_previous_role_available: bool
    reason: str
    role_switched: bool = False
    new_role: Optional[str] = None


class FeedbackResponse(BaseModel):
    user_id: str
    feedback_analysis: Dict[str, Any]


class HandsOnChatResponse(BaseModel):
    reply: str
    conversation_history: List[Dict[str, str]]
