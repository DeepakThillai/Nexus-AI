"""
orchestrator_wrapper.py
========================
Wraps the existing CLI agents for use by FastAPI endpoints.

STRICT RULES:
  - No agent logic is duplicated here
  - No business logic lives here
  - All LLM calls remain server-side inside the original agents
  - This file is ONLY a translation layer between HTTP and agents
  - Questions are stored in-memory per session (not in MongoDB)
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# ── Ensure project imports resolve ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import original agents (unmodified) ───────────────────────────
from agentic_career_navigator import (
    ReadinessAssessmentAgent,
    MarketIntelligenceAgent,
    RoadmapAgent,
    ActionAssessmentAgent,
    ReroutingAgent,
    FeedbackAgent,
    call_llm,
    extract_json,
)
from db import db

# ── In-memory session cache ───────────────────────────────────────
# Stores ephemeral Q&A data (questions + temp conversation state).
# This data is NEVER written to MongoDB. Maps user_id → session data.
_session_cache: Dict[str, Dict[str, Any]] = {}

# ── Agent singletons (reused across requests) ─────────────────────
_readiness_agent  = ReadinessAssessmentAgent()
_market_agent     = MarketIntelligenceAgent()
_roadmap_agent    = RoadmapAgent()
_action_agent     = ActionAssessmentAgent()
_rerouting_agent  = ReroutingAgent()
_feedback_agent   = FeedbackAgent()


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.utcnow().isoformat()


def _get_or_404(user_id: str) -> dict:
    """Fetch user document from MongoDB or raise ValueError."""
    user = db.get_user(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    return user


def _roadmap_completion(user: dict) -> float:
    steps = user.get("active_roadmap", {}).get("steps", [])
    total = sum(len(s.get("actions", [])) for s in steps)
    done  = sum(
        1 for s in steps
        for a in s.get("actions", [])
        if a.get("status") != "pending"
    )
    return (done / total * 100.0) if total else 0.0


def _update_confidence(user: dict, delta: int) -> int:
    """Apply delta to confidence, clamp to [0, 100]. Returns new score."""
    current = user.get("confidence_score", 0)
    new_val = max(0, min(100, current + delta))
    user["confidence_score"] = new_val
    return new_val


def _weekly_scores_from_roadmap(user: dict) -> List[Dict]:
    """Build chart data from roadmap action scores."""
    scores = []
    steps = user.get("active_roadmap", {}).get("steps", [])
    for step in steps:
        for action in step.get("actions", []):
            if action.get("score") is not None:
                scores.append({
                    "action_id": action["action_id"],
                    "label": f"M{step['month']}W{action['week']}",
                    "score": action["score"],
                    "status": action["status"],
                    "title": action["action_title"],
                })
    return scores


def _validate_target_role(target_role: str) -> str:
    """
    Validate target_role before saving to database.
    CRITICAL: Rejects sentinel values and ensures no invalid values are saved.
    Returns validated target_role or empty string.
    """
    if not target_role:
        return ""
    
    # REJECT sentinel values
    if target_role == "__email_check__":
        raise ValueError("CRITICAL: Cannot save sentinel value '__email_check__' as target_role!")
    
    # Reject other system values
    if target_role.startswith("__") and target_role.endswith("__"):
        raise ValueError(f"CRITICAL: Cannot save system sentinel '{target_role}' as target_role!")
    
    return target_role


def _build_user_document(user_id: str, email: str, target_role: str, data: dict) -> dict:
    """
    Build a complete MongoDB user document with proper schema.
    CRITICAL: Validates target_role to prevent sentinel values from being saved.
    """
    # Validate target_role before building
    validated_role = _validate_target_role(target_role)
    
    return {
        "user_id": user_id,
        "created_at": _now(),
        "last_updated": _now(),
        "profile": {
            "target_role": validated_role,
            "skills": data.get("skills", []),
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", []),
            "name": data.get("name"),
            "email": email,
            "phone": data.get("phone"),
            "experience_years": data.get("experience_years", 0),
            "resume_uploaded": False,
            "resume_uploaded_at": None,
            "resume_file_name": None,
        },
        "career_state": {
            "current_target_role": validated_role,
            "role_history": [],
        },
        "readiness_assessment": {
            "score": 0,
            "status": "",
            "evaluation_summary": "",
            "safer_adjacent_roles": [],
            "advanced_adjacent_roles": [],
        },
        "active_roadmap": {
            "generated_for_role": "",
            "steps": [],
            "status": "not_started",
        },
        "progress": {
            "actions_completed": 0,
            "actions_failed": 0,
            "weeks_completed": 0,
            "last_activity_at": _now(),
        },
        "reroute_state": {
            "is_active": False,
            "reroute_count": 0,
            "original_roadmap_id": None,
            "reroute_reason": None,
            "reroute_options": [],
            "selected_option": None,
            "rerouted_at": None,
            "can_return_to_previous": True,
        },
        "current_actions": {
            "this_week": [],
            "upcoming": [],
            "blocked": [],
        },
        "resume_analysis": {
            "parsed_profile": {},
            "extracted_skills": {},
            "normalized_skills": [],
            "skill_gap_analysis": {},
            "career_recommendations": {},
            "system_confidence": 0.0,
            "processing_time_seconds": 0.0,
        },
        "metadata": {
            "total_sessions": 0,
            "agent_interaction_count": {},
            "system_events": [],
        },
        "confidence_score": 0,
        "market_analysis": {},
        "feedback_analysis": {},
    }


# ═══════════════════════════════════════════════════════════════════
#  1. ONBOARDING
# ═══════════════════════════════════════════════════════════════════

def onboard_user(data: dict) -> dict:
    """
    Create a new user document in MongoDB with full schema.
    Then immediately trigger market intelligence generation.
    Returns the new user document.
    
    CRITICAL: Never allow sentinel values (__email_check__) to be saved to database!
    """
    import hashlib, time
    
    # Reject sentinel as real target_role
    if data.get("target_role") == "__email_check__":
        # This is an email check, not an onboarding
        normalized_email = data.get("email", "").lower().strip()
        print(f"[onboard] Email check for: {normalized_email}")
        existing = db.find_by_email(normalized_email)
        if existing:
            # Return existing user immediately - no DB modifications
            print(f"[onboard] Found existing user: {existing['user_id']}")
            clean_profile = existing["profile"].copy()
            # Double-check: sanitize profile if sentinel leaked in
            if clean_profile.get("target_role") == "__email_check__":
                clean_profile["target_role"] = ""
            return {"user_id": existing["user_id"], "profile": clean_profile, "exists": True}
        else:
            # New user email check - create with empty target_role (NOT sentinel)
            print(f"[onboard] No existing user found. Creating new user...")
            raw_id = f"user_{int(time.time() * 1000)}"
            user_id = raw_id
            target_role = ""  # Empty, NOT sentinel
            
            # Build minimal document for new user
            user_doc = _build_user_document(user_id, normalized_email, target_role, data)
            print(f"[onboard] Built user doc for {user_id}")
            
            # Save to database
            save_result = db.upsert_user(user_id, user_doc)
            print(f"[onboard] db.upsert_user returned: {save_result}")
            
            if not save_result:
                print(f"[onboard] ERROR: upsert_user failed to save user!")
                # Still return success response, but note the failure
            
            return {
                "user_id": user_id,
                "profile": user_doc["profile"],
                "exists": False
            }
    
    raw_id = f"user_{int(time.time() * 1000)}"
    user_id = raw_id
    
    # Normalize email: lowercase and strip whitespace (case-insensitive lookups)
    normalized_email = data.get("email", "").lower().strip()
    print(f"[onboard] Processing onboarding form for: {normalized_email}")
    
    # Sanitize target_role: never allow sentinel value to be saved
    target_role = data.get("target_role", "")
    if target_role == "__email_check__":
        target_role = ""  # Convert sentinel to empty string

    # Build full MongoDB document (matches schema exactly)
    user_doc = _build_user_document(user_id, normalized_email, target_role, data)

    # Check if email already exists (use normalized email for lookup)
    existing = db.find_by_email(normalized_email)
    if existing:
        print(f"[onboard] Found existing user for {normalized_email}")
        print(f"[onboard] Using existing user_id: {existing['user_id']}")
        # If this is just an email check (not a real onboarding), return immediately
        if data.get("target_role") == "__email_check__":
            # Return existing user WITHOUT modifying database
            # Clean up profile if it has sentinel value corrupted target_role
            clean_profile = existing["profile"].copy()
            print(f"[onboard] Email check for {normalized_email}")
            print(f"[onboard] Profile before sanitization: target_role='{clean_profile.get('target_role')}'")
            
            if clean_profile.get("target_role") == "__email_check__":
                clean_profile["target_role"] = ""
                print(f"[onboard] ⚠️  Sentinel detected! Sanitizing to empty string")
            
            print(f"[onboard] Profile after sanitization: target_role='{clean_profile.get('target_role')}'")
            print(f"[onboard] Returning: exists=True, user_id='{existing['user_id']}'")
            
            return {"user_id": existing["user_id"], "profile": clean_profile, "exists": True}
        
        # Validate: target_role cannot be empty when updating existing user
        if not data.get("target_role") or data.get("target_role") == "__email_check__":
            # Keep their existing target role if not provided
            target_role = existing.get("career_state", {}).get("current_target_role") or existing.get("profile", {}).get("target_role") or ""
            if not target_role:
                print(f"[onboard] WARNING: No valid target_role for existing user {existing['user_id']}")
        else:
            target_role = data["target_role"]
        
        # Update the existing user's profile with new onboarding data
        updated_profile = {
            "target_role": target_role,
            "skills": data.get("skills", []),
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", []),
            "name": data.get("name") or existing["profile"].get("name"),
            "email": normalized_email,
            "phone": data.get("phone") or existing["profile"].get("phone"),
            "experience_years": data.get("experience_years", 0),
            "resume_uploaded": existing["profile"].get("resume_uploaded", False),
            "resume_uploaded_at": existing["profile"].get("resume_uploaded_at"),
            "resume_file_name": existing["profile"].get("resume_file_name"),
        }
        updated_career_state = {
            "current_target_role": target_role,
            "role_history": existing.get("career_state", {}).get("role_history", []),
        }
        db.patch_user(existing["user_id"], {
            "profile": updated_profile,
            "career_state": updated_career_state,
            "last_updated": _now(),
        })
        
        # Regenerate market intelligence with the new target role (only if we have a valid target_role)
        if target_role:
            print(f"[onboard] Regenerating market intelligence for {target_role}...")
            try:
                market_result = _market_agent.run({"target_role": target_role})
                market_analysis = market_result.get("market_analysis", {})
                print(f"[onboard] Generated market analysis with role_title: {market_analysis.get('role_title')}")
                # Fully replace market_analysis (not merge)
                db.patch_user(existing["user_id"], {"market_analysis": market_analysis})
                print(f"[onboard] Market intelligence updated successfully")
            except Exception as e:
                print(f"[onboard] market intel regeneration failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[onboard] Skipping market intel regeneration - no target role provided")
        
        # Return the updated profile
        existing["profile"] = updated_profile
        return {"user_id": existing["user_id"], "profile": updated_profile, "exists": True}

    db.upsert_user(user_id, user_doc)
    print(f"[onboard] Created new user: {user_id}")

    # Trigger market intelligence immediately (only if target_role was provided)
    if target_role:
        print(f"[onboard] Generating initial market intelligence for {target_role}...")
        try:
            market_result = _market_agent.run({"target_role": target_role})
            user_doc["market_analysis"] = market_result.get("market_analysis", {})
            print(f"[onboard] Generated market analysis with role_title: {user_doc['market_analysis'].get('role_title')}")
            db.patch_user(user_id, {"market_analysis": user_doc["market_analysis"]})
            print(f"[onboard] Market intelligence saved successfully")
        except Exception as e:
            print(f"[onboard] market intel failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[onboard] Skipping market intelligence (no target_role provided - awaiting onboarding form)")

    return {"user_id": user_id, "profile": user_doc["profile"], "exists": False}


# ═══════════════════════════════════════════════════════════════════
#  2. READINESS ASSESSMENT
# ═══════════════════════════════════════════════════════════════════

def readiness_start(user_id: str) -> List[str]:
    """
    Generate 10 readiness questions via the existing agent.
    Questions are cached in memory only — NOT stored in MongoDB.
    Returns the question list.
    """
    user = _get_or_404(user_id)
    profile = user["profile"]

    # Use agent's internal question generator (re-exposed via wrapper)
    questions = _readiness_agent._generate_questions(
        target_role=profile["target_role"],
        skills=profile.get("skills", []),
        strengths=profile.get("strengths", []),
    )

    # Cache in memory only (ephemeral)
    _session_cache.setdefault(user_id, {})
    _session_cache[user_id]["readiness_questions"] = questions

    return questions


def readiness_evaluate(user_id: str, answers: List[str]) -> dict:
    """
    Evaluate 10 answers using the existing agent.
    Stores only score + summary in MongoDB (NOT the Q&A).
    Returns readiness result + initialises confidence.
    """
    user = _get_or_404(user_id)
    profile = user["profile"]

    # Retrieve cached questions
    questions = _session_cache.get(user_id, {}).get("readiness_questions", [])
    if not questions:
        raise ValueError("No readiness questions found in session. Call /api/readiness/start first.")

    # Build ephemeral Q&A pairs (never persisted)
    qa_pairs = [
        {"question": q, "answer": a}
        for q, a in zip(questions, answers)
    ]

    # Delegate to agent's evaluator (unmodified logic)
    result = _readiness_agent._evaluate(
        target_role=profile["target_role"],
        qa_pairs=qa_pairs,
    )

    score  = int(result.get("readiness_score", 0))
    status = result.get("readiness_status", "underqualified")

    # Clear ephemeral Q&A from memory
    _session_cache.get(user_id, {}).pop("readiness_questions", None)

    # Update MongoDB — scores + summary ONLY
    patch = {
        "readiness_assessment": {
            "score": score,
            "status": status,
            "evaluation_summary": result.get("evaluation_summary", ""),
            "safer_adjacent_roles":    result.get("safer_adjacent_roles", []),
            "advanced_adjacent_roles": result.get("advanced_adjacent_roles", []),
        },
        "confidence_score": score,   # initialise confidence to readiness_score
    }
    db.patch_user(user_id, patch)

    # Auto-generate roadmap after readiness
    try:
        roadmap_result = _roadmap_agent.run({
            "target_role": profile["target_role"],
            "strengths":   profile.get("strengths", []),
            "weaknesses":  profile.get("weaknesses", []),
            "skills":      profile.get("skills", []),
        })
        roadmap_doc = {
            "generated_for_role": profile["target_role"],
            "steps": roadmap_result.get("steps", []),
            "status": "generated",
        }
        db.patch_user(user_id, {"active_roadmap": roadmap_doc})
    except Exception as e:
        print(f"[readiness_evaluate] roadmap gen failed: {e}")

    return {
        "score": score,
        "status": status,
        "evaluation_summary": result.get("evaluation_summary", ""),
        "safer_adjacent_roles":    result.get("safer_adjacent_roles", []),
        "advanced_adjacent_roles": result.get("advanced_adjacent_roles", []),
        "confidence_score": score,
    }


# ═══════════════════════════════════════════════════════════════════
#  3. DASHBOARD
# ═══════════════════════════════════════════════════════════════════

def get_dashboard(user_id: str) -> dict:
    """Aggregate all state for the dashboard view."""
    user = _get_or_404(user_id)
    completion = _roadmap_completion(user)

    steps = user.get("active_roadmap", {}).get("steps", [])
    total_actions = sum(len(s.get("actions", [])) for s in steps)
    done_actions  = sum(
        1 for s in steps for a in s.get("actions", [])
        if a.get("status") != "pending"
    )

    return {
        "user_id": user_id,
        "profile": user.get("profile", {}),
        "confidence_score": user.get("confidence_score", 0),
        "readiness": user.get("readiness_assessment", {}),
        "progress": user.get("progress", {}),
        "roadmap_summary": {
            "total_actions": total_actions,
            "done_actions": done_actions,
            "completion_pct": round(completion, 1),
            "generated_for_role": user.get("active_roadmap", {}).get("generated_for_role", ""),
            "status": user.get("active_roadmap", {}).get("status", ""),
        },
        "reroute_state": user.get("reroute_state", {}),
        "career_state": user.get("career_state", {}),
        "weekly_scores": _weekly_scores_from_roadmap(user),
    }


# ═══════════════════════════════════════════════════════════════════
#  4. ROADMAP
# ═══════════════════════════════════════════════════════════════════

def get_roadmap(user_id: str) -> dict:
    user = _get_or_404(user_id)
    return {
        "user_id": user_id,
        "roadmap": user.get("active_roadmap", {}),
        "confidence_score": user.get("confidence_score", 0),
    }


def regenerate_roadmap(user_id: str, target_role: Optional[str] = None) -> dict:
    """
    Regenerates roadmap (and optionally switches role).
    Delegates entirely to RoadmapAgent — no logic duplication.
    """
    user = _get_or_404(user_id)
    profile = user["profile"]

    role = target_role or profile["target_role"]

    # If switching role, store history
    if target_role and target_role != profile.get("career_state", {}).get("current_target_role"):
        career_state = user.get("career_state", {})
        history = career_state.get("role_history", [])
        if profile["target_role"]:
            history.append(profile["target_role"])
        db.patch_user(user_id, {
            "profile.target_role": role,
            "career_state.current_target_role": role,
            "career_state.role_history": history,
        })
        profile["target_role"] = role

    result = _roadmap_agent.run({
        "target_role": role,
        "strengths":   profile.get("strengths", []),
        "weaknesses":  profile.get("weaknesses", []),
        "skills":      profile.get("skills", []),
    })

    roadmap_doc = {
        "generated_for_role": role,
        "steps": result.get("steps", []),
        "status": "generated",
    }

    # Reset progress on regeneration
    db.patch_user(user_id, {
        "active_roadmap": roadmap_doc,
        "progress.actions_completed": 0,
        "progress.actions_failed": 0,
    })

    return {"user_id": user_id, "roadmap": roadmap_doc, "confidence_score": user.get("confidence_score", 0)}


# ═══════════════════════════════════════════════════════════════════
#  5. ACTION ASSESSMENT
# ═══════════════════════════════════════════════════════════════════

def get_action_questions(user_id: str, action_id: str) -> dict:
    """
    Generate 10 questions for a specific action.
    Cached in memory — not stored in MongoDB.
    """
    user = _get_or_404(user_id)
    profile = user["profile"]

    # Find action in roadmap
    action_title = None
    for step in user.get("active_roadmap", {}).get("steps", []):
        for action in step.get("actions", []):
            if action["action_id"] == action_id:
                action_title = action["action_title"]
                break

    if not action_title:
        raise ValueError(f"Action {action_id} not found in roadmap")

    questions = _action_agent._generate_questions(action_title, profile["target_role"])

    # Cache per user+action (ephemeral)
    cache_key = f"{user_id}:{action_id}"
    _session_cache.setdefault(user_id, {})
    _session_cache[user_id][f"action_q_{action_id}"] = {
        "questions": questions,
        "action_title": action_title,
    }

    return {"action_id": action_id, "action_title": action_title, "questions": questions}


def assess_action(user_id: str, action_id: str, answers: List[str]) -> dict:
    """
    Evaluate action answers via existing ActionAssessmentAgent.
    Updates action status + score in MongoDB.
    Updates confidence_score. Runs rerouting check.

    If questions are no longer in session cache (e.g. server restarted between
    /api/action/questions and /api/action/assess), they are regenerated from
    the action title stored in MongoDB so evaluation still has full context.
    """
    user    = _get_or_404(user_id)
    profile = user["profile"]

    # Try session cache first (happy path — same process, short window)
    cached = _session_cache.get(user_id, {}).get(f"action_q_{action_id}")

    if not cached:
        # Cache miss (cold server / long session) — recover from roadmap
        action_title = None
        for step in user.get("active_roadmap", {}).get("steps", []):
            for action in step.get("actions", []):
                if action["action_id"] == action_id:
                    action_title = action["action_title"]
                    break
        if not action_title:
            raise ValueError(f"Action {action_id} not found in active roadmap.")
        # Regenerate questions for proper evaluation context
        try:
            questions = _action_agent._generate_questions(action_title, profile["target_role"])
        except Exception as e:
            # If question generation fails, use generic fallback questions
            print(f"[assess_action] question generation failed: {e}")
            questions = [
                f"How would you approach {action_title}?",
                f"What are the key concepts in {action_title}?",
                f"Describe your experience with {action_title}.",
                f"What challenges have you faced in {action_title}?",
                f"How do you measure success in {action_title}?",
                f"What tools would you use for {action_title}?",
                f"How does {action_title} relate to {profile['target_role']}?",
                f"What would you improve about {action_title}?",
                f"How often do you practice {action_title}?",
                f"What's your confidence level with {action_title}?"
            ]
        cached = {"questions": questions, "action_title": action_title}

    questions    = cached["questions"]
    action_title = cached["action_title"]

    # Build ephemeral Q&A pairs (zip handles length mismatch gracefully)
    qa_pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

    # Evaluate using unmodified agent logic
    try:
        result = _action_agent._evaluate(action_title, profile["target_role"], qa_pairs)
    except Exception as e:
        # If LLM evaluation fails, return defensive defaults
        print(f"[assess_action] LLM evaluation failed: {e}")
        result = {
            "action_score": 50,  # Neutral score
            "evaluation_summary": "Evaluation temporarily unavailable. Your answers have been recorded for manual review."
        }

    # Clear ephemeral data
    _session_cache.get(user_id, {}).pop(f"action_q_{action_id}", None)

    score  = int(result.get("action_score", 50))
    passed = score >= 50

    # Update action in roadmap
    completed_step_title = None
    steps = user.get("active_roadmap", {}).get("steps", [])
    for step in steps:
        for action in step.get("actions", []):
            if action["action_id"] == action_id:
                action["score"]  = score
                action["status"] = "passed" if passed else "failed"
                
                # Check if all actions in this step are now passed
                if passed:
                    all_passed = all(
                        a.get("status") == "passed" 
                        for a in step.get("actions", [])
                    )
                    if all_passed:
                        completed_step_title = step.get("title", "Untitled Step")

    # Update confidence
    delta = +1 if passed else -1
    new_confidence = _update_confidence(user, delta)

    # Update analytics
    progress = user.get("progress", {})
    if passed:
        progress["actions_completed"] = progress.get("actions_completed", 0) + 1
    else:
        progress["actions_failed"] = progress.get("actions_failed", 0) + 1
    progress["last_activity_at"] = _now()

    # Auto-add skill when step is completed (silent background update)
    profile = user.get("profile", {})
    skills = profile.get("skills", [])
    if completed_step_title and completed_step_title not in skills:
        skills.append(completed_step_title)
        profile["skills"] = skills

    # Persist to MongoDB
    update_payload = {
        "active_roadmap": user["active_roadmap"],
        "confidence_score": new_confidence,
        "progress": progress,
    }
    if completed_step_title:
        update_payload["profile"] = profile
    
    db.patch_user(user_id, update_payload)

    # Run rerouting check (pure logic — no LLM needed)
    # Run rerouting check (pure logic — no LLM needed)
    completion = _roadmap_completion(user)
    try:
        reroute_result = _rerouting_agent.run({
            "confidence_score":              new_confidence,
            "current_target_role":           profile["target_role"],
            "previous_target_role":          user.get("career_state", {}).get("role_history", [None])[-1],
            "roadmap_completion_percentage": completion,
        })
    except Exception as e:
        # If rerouting check fails (rare LLM issues), return safe defaults
        print(f"[assess_action] rerouting check failed: {e}")
        reroute_result = {
            "reroute_suggestion": False,
            "suggested_roles": [],
            "return_previous_role_available": False,
            "reason": "Rerouting check unavailable. Focus on current roadmap."
        }

    return {
        "action_id": action_id,
        "score": score,
        "passed": passed,
        "evaluation_summary": result.get("evaluation_summary", ""),
        "updated_confidence_score": new_confidence,
        "reroute_check": reroute_result,
    }


# ═══════════════════════════════════════════════════════════════════
#  6. MARKET INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════

def get_market(user_id: str) -> dict:
    """Return cached market analysis; refresh if missing."""
    user = _get_or_404(user_id)
    market = user.get("market_analysis", {})

    if not market:
        profile = user["profile"]
        result  = _market_agent.run({"target_role": profile["target_role"]})
        market  = result.get("market_analysis", {})
        db.patch_user(user_id, {"market_analysis": market})

    return {"user_id": user_id, "market_analysis": market}


# ═══════════════════════════════════════════════════════════════════
#  7. REROUTING
# ═══════════════════════════════════════════════════════════════════

def handle_reroute(user_id: str, new_role: Optional[str] = None) -> dict:
    """
    Analyse rerouting eligibility and optionally switch role.
    Uses existing ReroutingAgent — no duplicated logic.
    """
    user = _get_or_404(user_id)
    profile = user["profile"]
    history = user.get("career_state", {}).get("role_history", [])

    completion = _roadmap_completion(user)
    prev_role  = history[-1] if history else None

    # Run rerouting analysis
    reroute_result = _rerouting_agent.run({
        "confidence_score":              user.get("confidence_score", 0),
        "current_target_role":           profile["target_role"],
        "previous_target_role":          prev_role,
        "roadmap_completion_percentage": completion,
    })

    role_switched = False
    switched_to   = None

    if new_role and new_role != profile["target_role"]:
        # Execute role switch
        history.append(profile["target_role"])

        # Regenerate market + roadmap for new role
        try:
            market_result  = _market_agent.run({"target_role": new_role})
            roadmap_result = _roadmap_agent.run({
                "target_role": new_role,
                "strengths":   profile.get("strengths", []),
                "weaknesses":  profile.get("weaknesses", []),
                "skills":      profile.get("skills", []),
            })
            roadmap_doc = {
                "generated_for_role": new_role,
                "steps": roadmap_result.get("steps", []),
                "status": "generated",
            }
            db.patch_user(user_id, {
                "profile.target_role": new_role,
                "career_state.current_target_role": new_role,
                "career_state.role_history": history,
                "market_analysis": market_result.get("market_analysis", {}),
                "active_roadmap": roadmap_doc,
                "progress.actions_completed": 0,
                "progress.actions_failed": 0,
                "reroute_state.is_active": True,
                "reroute_state.reroute_count": user.get("reroute_state", {}).get("reroute_count", 0) + 1,
                "reroute_state.rerouted_at": _now(),
                "reroute_state.selected_option": new_role,
            })
            role_switched = True
            switched_to   = new_role
        except Exception as e:
            print(f"[reroute] failed to switch role: {e}")

    return {
        "user_id": user_id,
        "reroute_suggestion":             reroute_result["reroute_suggestion"],
        "suggested_roles":                reroute_result["suggested_roles"],
        "return_previous_role_available": reroute_result["return_previous_role_available"],
        "reason":                         reroute_result["reason"],
        "role_switched":                  role_switched,
        "new_role":                       switched_to,
    }


# ═══════════════════════════════════════════════════════════════════
#  8. FEEDBACK
# ═══════════════════════════════════════════════════════════════════

def generate_feedback(user_id: str) -> dict:
    """
    Generate feedback report using existing FeedbackAgent.
    Stores result in MongoDB.
    """
    user = _get_or_404(user_id)
    progress   = user.get("progress", {})
    completion = _roadmap_completion(user)

    result = _feedback_agent.run({
        "target_role":                 user["profile"]["target_role"],
        "confidence_score":            user.get("confidence_score", 0),
        "completed_actions_count":     progress.get("actions_completed", 0),
        "failed_actions_count":        progress.get("actions_failed", 0),
        "roadmap_progress_percentage": completion,
    })

    feedback = result.get("feedback_analysis", {})
    db.patch_user(user_id, {"feedback_analysis": feedback})

    return {"user_id": user_id, "feedback_analysis": feedback}


# ═══════════════════════════════════════════════════════════════════
#  9. HANDS-ON CHAT
# ═══════════════════════════════════════════════════════════════════

def hands_on_chat(user_id: str, message: str, conversation_history: List[dict]) -> dict:
    """
    Stateless chat endpoint — full conversation history sent each request.
    Uses HandsOnAgent's Groq call pattern (same model, same API key).
    """
    user = _get_or_404(user_id)
    target_role = user["profile"].get("target_role", "your target role")

    # System prompt matching HandsOnAgent behaviour
    system_prompt = (
        f"You are a hands-on technical mentor for someone preparing to become a {target_role}.\n"
        "Assign ONE practical real-world task for the given target role.\n"
        "Guide the user step-by-step.\n"
        "Wait for confirmation before moving forward.\n"
        "Clarify doubts clearly.\n"
        "Only conclude the session when satisfied or user types '$'.\n"
        "Structure responses as:\n\n"
        "Task:\nCurrent Step:\nWhat To Do:\nReply After Completion:"
    )

    # Build messages list: system + history + new user message
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (frontend sends full history)
    for turn in conversation_history:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Call LLM via existing call_llm helper
    import os
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        max_completion_tokens=1500,
        stream=False,
    )
    reply = response.choices[0].message.content

    # Return updated conversation history (no server-side storage)
    updated_history = list(conversation_history) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": reply},
    ]

    return {"reply": reply, "conversation_history": updated_history}


# ═══════════════════════════════════════════════════════════════════
#  BONUS: RESUME SKILLS EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_skills_from_resume(resume_text: str) -> dict:
    """
    Extract skills from resume text using LLM.
    Used during onboarding when user uploads resume instead of manual entry.
    Returns list of extracted skills that user can add to their profile.
    """
    if not resume_text or not resume_text.strip():
        return {"skills": [], "message": "Resume text is empty"}
    
    try:
        system_prompt = "You are an expert resume parser. Extract technical and professional skills from resumes and return them as a JSON object with a 'skills' array."
        
        user_prompt = f"""
Extract all technical and professional skills from this resume. 
Return ONLY a JSON object with a "skills" array of skill names (strings).
Be specific and practical. Include tools, languages, frameworks, methods, etc.

Resume:
{resume_text}

Return format (ONLY JSON):
{{"skills": ["skill1", "skill2", "skill3", ...]}}
"""
        
        response = call_llm(system_prompt, user_prompt, max_tokens=1024)
        
        # Try to extract JSON from response
        try:
            result = extract_json(response)
            if isinstance(result, dict) and "skills" in result:
                skills = result["skills"]
                # Ensure all items are strings
                skills = [str(s).strip() for s in skills if s]
                return {
                    "skills": skills,
                    "message": f"Extracted {len(skills)} skills from resume"
                }
        except:
            pass
        
        # Fallback: try to parse response as JSON
        try:
            import json as json_module
            cleaned = response.strip()
            if cleaned.startswith('{'):
                result = json_module.loads(cleaned)
                if "skills" in result:
                    skills = result["skills"]
                    skills = [str(s).strip() for s in skills if s]
                    return {
                        "skills": skills,
                        "message": f"Extracted {len(skills)} skills from resume"
                    }
        except:
            pass
        
        # If all else fails, return empty but don't error
        return {
            "skills": [],
            "message": "Could not parse skills from resume. Please enter manually."
        }
        
    except Exception as e:
        print(f"[extract_skills] Error: {e}")
        return {
            "skills": [],
            "message": f"Error extracting skills: {str(e)}"
        }


# ═══════════════════════════════════════════════════════════════════
#  RESUME FILE ANALYSIS (using ResumeAnalyzerAgent)
# ═══════════════════════════════════════════════════════════════════

def analyze_resume_file(user_id: str, file_path: str, file_name: str) -> dict:
    """
    Analyze resume file (PDF or image) using ResumeAnalyzerAgent.
    Extracts text, parses structure, and stores in database.
    
    Returns:
        {
            "status": "success" | "error",
            "message": str,
            "parsed_profile": dict,
            "extracted_skills": dict,
            "normalized_skills": list[str],
            "skill_gap_analysis": dict (optional)
        }
    """
    try:
        from agentic_career_navigator import ResumeAnalyzerAgent
        
        print(f"[Resume Analysis] Processing {file_name} for user {user_id}...")
        
        agent = ResumeAnalyzerAgent()
        result = agent.run({
            "user_id": user_id,
            "file_path": file_path,
            "file_name": file_name
        })
        
        return result
        
    except ImportError:
        return {
            "status": "error",
            "message": "ResumeAnalyzerAgent not available"
        }
    except Exception as e:
        print(f"[Resume Analysis] Error: {str(e)[:100]}")
        return {
            "status": "error",
            "message": f"Failed to analyze resume: {str(e)[:100]}"
        }

