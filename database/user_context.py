"""
User Context Manager

Manages persistent storage and retrieval of all user-related data
across agent interactions. Stores contexts as JSON files with MongoDB
optional sync.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class UserContextManager:
    """
    Centralized manager for user context that persists across sessions.
    All agents must read from and write to this context.
    
    Storage: JSON files in data/user_contexts/
    Optional: MongoDB sync via MongoStore
    """
    
    def __init__(self, context_dir: str = None):
        """
        Initialize context manager
        
        Args:
            context_dir: Directory to store user context files (default: data/user_contexts)
        """
        if context_dir is None:
            # Default to data/user_contexts folder
            context_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "user_contexts"
            )
        
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)
    
    def get_context_path(self, user_id: str) -> Path:
        """Get file path for user context"""
        return self.context_dir / f"{user_id}_context.json"
    
    def initialize_context(self, user_id: str) -> Dict[str, Any]:
        """
        Initialize a new user context with complete schema
        
        Returns:
            Fresh user context structure
        """
        context = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            
            # === PROFILE === (populated from resume or manual input)
            "profile": {
                "name": None,
                "email": None,
                "phone": None,
                "education": {
                    "degree": None,
                    "institution": None,
                    "graduation_year": None,
                    "cgpa": None
                },
                "experience_years": 0,
                "skills": {
                    "technical": [],
                    "frameworks": [],
                    "databases": [],
                    "tools": [],
                    "soft_skills": []
                },
                "projects": [],
                "certifications": [],
                "achievements": [],
                "languages": [],
                "linkedin": None,
                "github": None,
                "resume_uploaded": False,
                "resume_uploaded_at": None,
                "resume_file_name": None
            },
            
            # === CAREER STATE ===
            "career_state": {
                "current_target_role": None,
                "original_target_role": None,
                "role_tier": None,
                "role_history": []
            },
            
            # === READINESS ASSESSMENT ===
            "readiness_assessment": {
                "status": None,
                "confidence_score": 0.0,
                "skill_match_percentage": 0,
                "matched_skills": [],
                "missing_skills": [],
                "surplus_skills": [],
                "recommendation": None,
                "reasoning": None,
                "last_assessed_at": None,
                "assessment_history": []
            },
            
            # === ACTIVE ROADMAP ===
            "active_roadmap": {
                "roadmap_id": None,
                "created_at": None,
                "duration_months": 5,
                "total_weeks": 20,
                "current_phase": None,
                "current_week": None,
                "completion_percentage": 0,
                "phases": [],
                "status": "not_started"
            },
            
            # === PROGRESS TRACKING ===
            "progress": {
                "weeks_completed": 0,
                "actions_completed": 0,
                "actions_failed": 0,
                "current_streak_weeks": 0,
                "total_hours_invested": 0,
                "last_activity_at": None,
                "weekly_engagement": {}
            },
            
            # === REROUTING STATE ===
            "reroute_state": {
                "is_active": False,
                "reroute_count": 0,
                "original_roadmap_id": None,
                "reroute_reason": None,
                "reroute_options": [],
                "selected_option": None,
                "rerouted_at": None,
                "can_return_to_previous": True
            },
            
            # === CURRENT ACTIONS ===
            "current_actions": {
                "this_week": [],
                "upcoming": [],
                "blocked": []
            },
            
            # === RESUME ANALYSIS ===
            "resume_analysis": {
                "parsed_profile": {},
                "extracted_skills": {},
                "normalized_skills": [],
                "skill_gap_analysis": {},
                "career_recommendations": {},
                "system_confidence": 0.0,
                "processing_time_seconds": 0.0
            },
            
            # === METADATA ===
            "metadata": {
                "total_sessions": 0,
                "agent_interaction_count": {},
                "system_events": []
            }
        }
        
        self.save_context(user_id, context)
        return context
    
    def load_context(self, user_id: str) -> Dict[str, Any]:
        """
        Load user context from storage
        
        Returns:
            User context dict or creates new if doesn't exist
        """
        context_path = self.get_context_path(user_id)
        
        if context_path.exists():
            with open(context_path, 'r') as f:
                context = json.load(f)
            return context
        else:
            return self.initialize_context(user_id)
    
    def save_context(self, user_id: str, context: Dict[str, Any]) -> None:
        """Save user context to JSON file"""
        context["last_updated"] = datetime.now().isoformat()
        context_path = self.get_context_path(user_id)
        
        with open(context_path, 'w') as f:
            json.dump(context, f, indent=2, default=str)
    
    def update_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Update profile section"""
        context = self.load_context(user_id)
        context["profile"].update(profile_data)
        self.save_context(user_id, context)
    
    def update_career_state(self, user_id: str, career_data: Dict[str, Any]) -> None:
        """Update career state section"""
        context = self.load_context(user_id)
        context["career_state"].update(career_data)
        self.save_context(user_id, context)
    
    def update_readiness(self, user_id: str, readiness_data: Dict[str, Any]) -> None:
        """Update readiness assessment"""
        context = self.load_context(user_id)
        context["readiness_assessment"].update(readiness_data)
        self.save_context(user_id, context)
    
    def update_roadmap(self, user_id: str, roadmap_data: Dict[str, Any]) -> None:
        """Update active roadmap"""
        context = self.load_context(user_id)
        context["active_roadmap"].update(roadmap_data)
        self.save_context(user_id, context)
    
    def update_resume_analysis(self, user_id: str, analysis_data: Dict[str, Any]) -> None:
        """Update resume analysis results"""
        context = self.load_context(user_id)
        context["resume_analysis"].update(analysis_data)
        self.save_context(user_id, context)
    
    def log_event(self, user_id: str, event_type: str, agent_name: str = None, details: Dict = None) -> None:
        """Log a system event"""
        context = self.load_context(user_id)
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "agent": agent_name,
            "details": details or {}
        }
        
        context["metadata"]["system_events"].append(event)
        
        # Keep only last 100 events
        if len(context["metadata"]["system_events"]) > 100:
            context["metadata"]["system_events"] = context["metadata"]["system_events"][-100:]
        
        self.save_context(user_id, context)
    
    def get_full_context(self, user_id: str) -> Dict[str, Any]:
        """Get complete user context"""
        return self.load_context(user_id)
    
    def export_context(self, user_id: str, export_path: str = None) -> str:
        """Export user context to JSON file"""
        context = self.load_context(user_id)
        
        if not export_path:
            export_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "exports"
            )
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(
                export_dir,
                f"{user_id}_context_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(export_path, 'w') as f:
            json.dump(context, f, indent=2, default=str)
        
        return export_path
    
    def clear_context(self, user_id: str) -> None:
        """Clear user context (use with caution)"""
        context_path = self.get_context_path(user_id)
        if context_path.exists():
            context_path.unlink()
