"""
╔══════════════════════════════════════════════════════════════════╗
║           AGENTIC CAREER NAVIGATOR                               ║
║           Powered by Groq (openai/gpt-oss-120b)                  ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
  Orchestrator → coordinates all agents and maintains persistent state
  ├── ResumeAnalyzerAgent       → extracts skills & profile from resumes
  ├── ReadinessAssessmentAgent  → evaluates user readiness for target role
  ├── MarketIntelligenceAgent   → generates structured market analysis
  ├── RoadmapAgent              → builds a 5-month action roadmap
  ├── ActionAssessmentAgent     → evaluates mastery of individual actions
  ├── ReroutingAgent            → handles confidence-based role switching
  └── FeedbackAgent             → generates comprehensive progress feedback

Usage:
  python agentic_career_navigator.py
"""

import json
import re
import os
import sys
import time
import shutil
from datetime import date, timedelta, datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add project root to path so we can import database modules
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import AuthenticationError, Groq

# Import database modules
from database.user_context import UserContextManager
from database.mongo_store import MongoStore

# Import resume analysis dependencies
try:
    from pdfminer.high_level import extract_text
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
#  GROQ CLIENT + SHARED LLM CALL
# ═══════════════════════════════════════════════════════════════════

API_ENV_VAR = "GROQ_API_KEY"
MODEL = "openai/gpt-oss-120b"  # Groq-hosted GPT-style model

_client: Optional[Groq] = None


def _get_api_key() -> str:
    """Read and validate the Groq API key from environment."""
    key = os.environ.get(API_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing {API_ENV_VAR}. Set it in .env file or as environment variable."
        )
    return key


def get_client() -> Groq:
    """Lazily initialize and return the shared Groq client."""
    global _client
    if _client is None:
        api_key = _get_api_key()
        _client = Groq(api_key=api_key)
    return _client


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
    """
    Central function for ALL LLM calls in the system.
    Every agent must use this — no direct Groq calls elsewhere.

    Returns the raw string content from the model.
    """
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
            stream=False,
        )
    except AuthenticationError as exc:
        raise RuntimeError(
            f"Groq authentication failed. Check {API_ENV_VAR} and use a valid key."
        ) from exc
    return response.choices[0].message.content.strip()


def extract_json(raw: str) -> dict:
    """
    Robustly parse JSON from a model response that may contain
    markdown fences, prose preambles, or trailing text.
    """
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{raw[:500]}")


# ═══════════════════════════════════════════════════════════════════
#  PRETTY PRINTER UTILITY
# ═══════════════════════════════════════════════════════════════════

def print_section(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_dict(d: dict, indent: int = 2) -> None:
    """Print a dict as formatted JSON for clean CLI output."""
    print(json.dumps(d, indent=indent, default=str))


# ═══════════════════════════════════════════════════════════════════
#  AGENT 0 — ResumeAnalyzerAgent
# ═══════════════════════════════════════════════════════════════════

class ResumeAnalyzerAgent:
    """
    Analyzes resume files to extract skills, experience, education.
    Performs skill gap analysis against target role requirements.
    
    INPUT:
        user_id: str
        file_path: str (path to resume file)
        file_name: str (original filename)
    
    OUTPUT:
        parsed_profile: dict
        extracted_skills: dict (categorized)
        normalized_skills: list
        skill_gap_analysis: dict
        career_recommendations: dict
    """
    
    def __init__(self):
        self.name = "ResumeAnalyzerAgent"
        self.context_manager = UserContextManager()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfminer.six"""
        if not _PDF_AVAILABLE:
            return ""
        try:
            text = extract_text(pdf_path)
            return text.strip()
        except Exception as e:
            print(f"  [Resume] PDF extraction error: {str(e)[:50]}")
            return ""

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using Tesseract OCR"""
        if not _OCR_AVAILABLE:
            return ""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            print(f"  [Resume] OCR extraction error: {str(e)[:50]}")
            return ""

    def extract_text(self, file_path: str) -> tuple:
        """Extract text from file (PDF or image)"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
            return text, "pdf"
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            text = self.extract_text_from_image(file_path)
            return text, "ocr"
        else:
            return "", "unsupported"

    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Parse resume text using LLM to extract structured information"""
        if len(resume_text) < 50:
            return {
                "parsed_profile": {},
                "extracted_skills": {
                    "programming_languages": [],
                    "frameworks": [],
                    "databases": [],
                    "cloud_platforms": [],
                    "tools": [],
                    "soft_skills": []
                },
                "projects": [],
                "achievements": [],
                "languages": []
            }

        system = (
            "You are a professional resume parser. Extract ALL relevant information. "
            "Return ONLY valid JSON matching the provided schema. No explanations."
        )

        schema = {
            "parsed_profile": {
                "name": "string or null",
                "email": "string or null",
                "phone": "string or null",
                "location": "string or null",
                "linkedin": "string or null",
                "github": "string or null",
                "experience_years": 0,
                "job_titles": [],
                "education": [],
                "certifications": []
            },
            "extracted_skills": {
                "programming_languages": [],
                "frameworks": [],
                "databases": [],
                "cloud_platforms": [],
                "tools": [],
                "soft_skills": []
            },
            "projects": [{"name": "string", "description": "string", "technologies": []}],
            "achievements": [],
            "languages": []
        }

        user = (
            f"Extract information from this resume:\n\n{resume_text[:3000]}\n\n"
            f"Return this exact JSON schema:\n{json.dumps(schema, indent=2)}"
        )

        try:
            response = call_llm(system, user, max_tokens=2500)
            response = response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(response)
            return parsed
        except Exception as e:
            print(f"  [Resume] JSON parse error: {str(e)[:50]}")
            return {
                "parsed_profile": {},
                "extracted_skills": {
                    "programming_languages": [],
                    "frameworks": [],
                    "databases": [],
                    "cloud_platforms": [],
                    "tools": [],
                    "soft_skills": []
                },
                "projects": [],
                "achievements": [],
                "languages": []
            }

    def normalize_skills(self, extracted_skills: Dict[str, List[str]]) -> List[str]:
        """Flatten and normalize skills into single list"""
        all_skills = []
        for category, skills in extracted_skills.items():
            if isinstance(skills, list):
                all_skills.extend(skills)
        
        normalized = list(set([s.strip().lower() for s in all_skills if s]))
        return sorted(normalized)

    def calculate_skill_gap(self, user_id: str, resume_skills: List[str]) -> Dict[str, Any]:
        """Calculate skill gap between resume and target role"""
        context = self.context_manager.load_context(user_id)
        
        # Get required skills (support multiple schema formats)
        required_skills = context.get("career_state", {}).get("required_skills", [])
        if not required_skills:
            required_skills = context.get("required_skills", [])
        
        # Normalize
        required_normalized = [s.strip().lower() for s in required_skills if s]
        resume_normalized = [s.strip().lower() for s in resume_skills if s]
        
        # Find matches
        matched_skills = []
        missing_skills = []
        
        for req_skill in required_normalized:
            found = False
            for res_skill in resume_normalized:
                if req_skill in res_skill or res_skill in req_skill:
                    matched_skills.append(req_skill)
                    found = True
                    break
            if not found:
                missing_skills.append(req_skill)
        
        # Calculate percentages
        total_required = len(required_normalized) if required_normalized else 1
        matched_count = len(matched_skills)
        missing_count = len(missing_skills)
        
        match_percentage = (matched_count / total_required * 100) if total_required > 0 else 0
        gap_percentage = 100 - match_percentage
        
        return {
            "required_skills_count": total_required,
            "user_skills_count": matched_count,
            "skill_gap_count": missing_count,
            "skill_gap_percentage": round(gap_percentage, 1),
            "match_percentage": round(match_percentage, 1),
            "matched_skills": matched_skills[:10],  # Top 10
            "missing_skills": missing_skills[:5],   # Top 5
            "high_impact_missing": missing_skills[:3]  # Top 3 critical
        }

    def run(self, input_data: dict) -> dict:
        """Complete resume analysis workflow - Extract and normalize skills only (gap analysis happens during readiness)"""
        user_id = input_data["user_id"]
        file_path = input_data["file_path"]
        file_name = input_data.get("file_name", "resume")
        
        start_time = time.time()
        
        try:
            print(f"  [1/3] Extracting text from {file_name}...")
            raw_text, extraction_method = self.extract_text(file_path)
            
            if not raw_text or len(raw_text) < 50:
                return {"status": "error", "message": "Could not extract text from resume"}
            
            print(f"  [2/3] Parsing resume structure...")
            parsed_data = self.parse_resume(raw_text)
            
            print(f"  [3/3] Normalizing skills...")
            extracted_skills_dict = parsed_data.get("extracted_skills", {})
            normalized_skills = self.normalize_skills(extracted_skills_dict)
            
            # NOTE: Skill gap analysis moved to ReadinessAssessmentAgent
            # Gap analysis requires knowing target_role, which is selected AFTER resume upload
            
            processing_time = time.time() - start_time
            
            # Update context with resume data
            context = self.context_manager.load_context(user_id)
            parsed_profile = parsed_data.get("parsed_profile", {})
            
            # Map to profile section
            context["profile"].update({
                "name": parsed_profile.get("name"),
                "email": parsed_profile.get("email"),
                "phone": parsed_profile.get("phone"),
                "experience_years": parsed_profile.get("experience_years", 0),
                "resume_uploaded": True,
                "resume_uploaded_at": datetime.now().isoformat(),
                "resume_file_name": file_name
            })
            
            # Map education
            education_list = parsed_profile.get("education", [])
            if education_list and len(education_list) > 0:
                edu_str = education_list[0] if isinstance(education_list[0], str) else ""
                if edu_str:
                    context["profile"]["education"]["degree"] = edu_str
            
            # Map skills
            if extracted_skills_dict:
                context["profile"]["skills"] = {
                    "technical": extracted_skills_dict.get("programming_languages", []),
                    "frameworks": extracted_skills_dict.get("frameworks", []),
                    "databases": extracted_skills_dict.get("databases", []),
                    "tools": extracted_skills_dict.get("tools", []) + extracted_skills_dict.get("cloud_platforms", []),
                    "soft_skills": extracted_skills_dict.get("soft_skills", [])
                }
            
            # Map other fields
            context["profile"]["projects"] = parsed_data.get("projects", [])
            context["profile"]["certifications"] = parsed_data.get("certifications", []) or parsed_profile.get("certifications", [])
            context["profile"]["achievements"] = parsed_data.get("achievements", [])
            context["profile"]["languages"] = parsed_data.get("languages", [])
            context["profile"]["linkedin"] = parsed_profile.get("linkedin")
            context["profile"]["github"] = parsed_profile.get("github")
            
            # Note: Skill gap analysis will be done during readiness assessment
            # after target role is selected
            
            # Store analysis
            context["resume_analysis"] = {
                "parsed_profile": parsed_profile,
                "extracted_skills": extracted_skills_dict,
                "normalized_skills": normalized_skills,
                "extraction_method": extraction_method,
                "processing_time_seconds": round(processing_time, 2)
            }
            
            self.context_manager.save_context(user_id, context)
            
            # Optionally sync to MongoDB
            try:
                mongo = MongoStore()
                mongo.sync(user_id, context)
            except:
                pass  # MongoDB optional
            
            print(f"  ✓ Resume analysis complete in {processing_time:.2f}s")
            print(f"  ✓ {len(normalized_skills)} skills extracted")
            
            return {
                "status": "success",
                "parsed_profile": parsed_profile,
                "extracted_skills": extracted_skills_dict,
                "normalized_skills": normalized_skills
            }
            
        except Exception as e:
            print(f"  ✗ Resume analysis failed: {str(e)[:100]}")
            return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  AGENT 1 — ReadinessAssessmentAgent
# ═══════════════════════════════════════════════════════════════════

class ReadinessAssessmentAgent:
    """
    Evaluates the user's readiness for their target role.

    INPUT:
        target_role : str
        skills      : list[str]
        strengths   : list[str]

    OUTPUT:
        readiness_score        : int  (0-100)
        readiness_status       : str  ("underqualified"|"qualified"|"overqualified")
        evaluation_summary     : str
        safer_adjacent_roles   : list[str]
        advanced_adjacent_roles: list[str]

    Note: Q&A is kept in-memory only — NOT stored in persistent state.
    """

    def __init__(self):
        self.name = "ReadinessAssessmentAgent"

    def _generate_questions(self, target_role: str, skills: list, strengths: list) -> list[str]:
        """Ask the LLM to produce 10 readiness-evaluation questions."""
        system = (
            "You are an expert career assessment interviewer. "
            "Generate exactly 10 concise, specific questions to evaluate a candidate's "
            "readiness for the given role. Return ONLY a JSON array of 10 question strings. "
            "No preamble, no numbering outside the array."
        )
        user = (
            f"Target Role: {target_role}\n"
            f"Candidate Skills: {', '.join(skills)}\n"
            f"Candidate Strengths: {', '.join(strengths)}\n\n"
            "Return format: [\"question1\", \"question2\", ..., \"question10\"]"
        )
        raw = call_llm(system, user, max_tokens=800)

        # Parse array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Fallback: split numbered lines
        lines = [l.strip().lstrip("0123456789.)- ") for l in raw.splitlines() if l.strip()]
        return lines[:10] if len(lines) >= 10 else lines

    def _collect_answers(self, questions: list[str]) -> list[dict]:
        """Collect answers from the user via CLI. Q&A stays in memory only."""
        qa_pairs = []
        print("\n  Please answer each question honestly (press Enter to submit each answer):\n")
        for i, q in enumerate(questions, 1):
            print(f"  Q{i}: {q}")
            answer = input("  Your answer: ").strip()
            qa_pairs.append({"question": q, "answer": answer})
            print()
        return qa_pairs  # ephemeral — not written to persistent state

    def _evaluate(self, target_role: str, qa_pairs: list[dict]) -> dict:
        """Send Q&A to LLM for evaluation. Returns structured result."""
        system = (
            "You are a senior career evaluation AI. "
            "Evaluate the candidate's readiness based on their Q&A responses. "
            "Return ONLY valid JSON matching the exact schema provided."
        )
        qa_text = "\n".join(
            f"Q: {p['question']}\nA: {p['answer']}" for p in qa_pairs
        )
        schema = json.dumps({
            "readiness_score": "<int 0-100>",
            "readiness_status": "<'underqualified'|'qualified'|'overqualified'>",
            "evaluation_summary": "<string>",
            "safer_adjacent_roles": ["<role1>", "<role2>", "<role3>"],
            "advanced_adjacent_roles": ["<role1>", "<role2>"]
        }, indent=2)
        user = (
            f"Target Role: {target_role}\n\n"
            f"Q&A Session:\n{qa_text}\n\n"
            f"Return this exact JSON schema filled in:\n{schema}"
        )
        raw = call_llm(system, user, max_tokens=1000)
        return extract_json(raw)

    def run(self, input_data: dict) -> dict:
        """
        Orchestrator calls this. Drives the full assessment flow.
        Returns output schema dict (scores + summaries only — no Q&A).
        """
        target_role = input_data["target_role"]
        skills      = input_data.get("skills", [])
        strengths   = input_data.get("strengths", [])

        print_section(f"READINESS ASSESSMENT — {target_role}")

        # Step 1: GPT generates questions
        print("  [1/3] Generating assessment questions via GPT...")
        questions = self._generate_questions(target_role, skills, strengths)
        print(f"  ✓ {len(questions)} questions generated.\n")

        # Step 2: Collect answers (Q&A stays in memory only)
        qa_pairs = self._collect_answers(questions)

        # Step 3: GPT evaluates responses
        print("  [3/3] Evaluating responses via GPT...")
        result = self._evaluate(target_role, qa_pairs)
        # qa_pairs goes out of scope here — never persisted

        print("\n  ✓ Assessment complete.")
        return result   # output schema only


# ═══════════════════════════════════════════════════════════════════
#  AGENT 2 — MarketIntelligenceAgent
# ═══════════════════════════════════════════════════════════════════

class MarketIntelligenceAgent:
    """
    Generates structured market intelligence for the target role.

    INPUT:
        target_role: str

    OUTPUT: dict matching the market_analysis schema exactly.
    """

    def __init__(self):
        self.name = "MarketIntelligenceAgent"

    def run(self, input_data: dict) -> dict:
        target_role = input_data["target_role"]
        today       = date.today().isoformat()

        print_section(f"MARKET INTELLIGENCE — {target_role}")
        print("  Generating market analysis via GPT...")

        system = (
            "You are a senior labor market analyst with real-time industry data. "
            "Return ONLY valid JSON. No prose, no markdown fences."
        )
        schema = {
            "market_analysis": {
                "role_title": target_role,
                "demand_score": "<int 0-100>",
                "competition_level": "<low|medium|high>",
                "entry_barrier": "<low|medium|high>",
                "market_trend": "<growing|stable|declining>",
                "avg_salary_range_usd": "<e.g. 80k-120k>",
                "required_experience_years": "<e.g. 0-2>",
                "key_hiring_companies": ["<Company1>", "<Company2>", "<Company3>"],
                "in_demand_skills": ["<skill1>", "<skill2>", "<skill3>"],
                "market_saturation": "<low|medium|high>",
                "job_availability": "<abundant|moderate|scarce>",
                "adjacent_safer_roles": [
                    {
                        "role": "<Role Name>",
                        "reason": "<Why safer>",
                        "demand_score": "<int 0-100>",
                        "entry_barrier": "<low|medium|high>"
                    }
                ],
                "market_notes": "<Key insights about this role's market>",
                "last_updated": today
            }
        }
        user = (
            f"Generate a complete, realistic market intelligence report for: {target_role}\n\n"
            f"Fill in this exact JSON schema:\n{json.dumps(schema, indent=2)}"
        )
        raw    = call_llm(system, user, max_tokens=1500)
        result = extract_json(raw)

        # Guarantee last_updated is today (model might hallucinate old date)
        result["market_analysis"]["last_updated"] = today

        print("  ✓ Market analysis complete.")
        return result


# ═══════════════════════════════════════════════════════════════════
#  AGENT 3 — RoadmapAgent
# ═══════════════════════════════════════════════════════════════════

class RoadmapAgent:
    """
    Generates a 5-month structured roadmap.

    INPUT:
        target_role: str
        strengths  : list
        weaknesses : list
        skills     : list

    OUTPUT: { generated_for_role: str, steps: [...] }
            Exactly 5 months × 4 actions each.
    """

    def __init__(self):
        self.name = "RoadmapAgent"

    def run(self, input_data: dict) -> dict:
        target_role = input_data["target_role"]
        strengths   = input_data.get("strengths", [])
        weaknesses  = input_data.get("weaknesses", [])
        skills      = input_data.get("skills", [])

        print_section(f"ROADMAP GENERATION — {target_role}")
        print("  Building 5-month roadmap via GPT...")

        system = (
            "You are an expert career roadmap architect. "
            "Return ONLY valid JSON. No prose, no markdown fences. "
            "STRICT REQUIREMENT: Exactly 5 steps (months), each with exactly 4 actions."
        )

        schema = {
            "steps": [
                {
                    "month": 1,
                    "step_title": "<Month 1 focus title>",
                    "actions": [
                        {"action_id": "action_1",  "week": 1, "action_title": "<Action title>", "status": "pending", "score": None},
                        {"action_id": "action_2",  "week": 2, "action_title": "<Action title>", "status": "pending", "score": None},
                        {"action_id": "action_3",  "week": 3, "action_title": "<Action title>", "status": "pending", "score": None},
                        {"action_id": "action_4",  "week": 4, "action_title": "<Action title>", "status": "pending", "score": None},
                    ]
                },
                "... repeat for months 2, 3, 4, 5 with action_ids action_5..action_20"
            ]
        }

        user = (
            f"Target Role    : {target_role}\n"
            f"Strengths      : {', '.join(strengths)}\n"
            f"Weaknesses     : {', '.join(weaknesses)}\n"
            f"Current Skills : {', '.join(skills)}\n\n"
            "Generate a realistic, progressive 5-month career preparation roadmap.\n"
            "Each month should build on the previous. Action IDs must be:\n"
            "  Month 1: action_1  to action_4\n"
            "  Month 2: action_5  to action_8\n"
            "  Month 3: action_9  to action_12\n"
            "  Month 4: action_13 to action_16\n"
            "  Month 5: action_17 to action_20\n\n"
            f"Return this exact JSON structure:\n{json.dumps(schema, indent=2)}"
        )

        raw    = call_llm(system, user, max_tokens=2500)
        result = extract_json(raw)

        # Validate and patch structure
        steps = result.get("steps", [])
        if len(steps) != 5:
            raise ValueError(f"RoadmapAgent expected 5 steps, got {len(steps)}")

        for month_idx, step in enumerate(steps, 1):
            step["month"] = month_idx
            actions = step.get("actions", [])
            if len(actions) != 4:
                raise ValueError(f"Month {month_idx} must have exactly 4 actions, got {len(actions)}")
            for week_idx, action in enumerate(actions, 1):
                base = (month_idx - 1) * 4 + week_idx
                action["action_id"] = f"action_{base}"
                action["week"]      = week_idx
                action.setdefault("status", "pending")
                action.setdefault("score", None)

        print("  ✓ Roadmap generated (5 months × 4 actions).")
        return {"generated_for_role": target_role, "steps": steps}


# ═══════════════════════════════════════════════════════════════════
#  AGENT 4 — ActionAssessmentAgent
# ═══════════════════════════════════════════════════════════════════

class ActionAssessmentAgent:
    """
    Evaluates mastery of a single roadmap action.

    INPUT:
        action_id   : str
        action_title: str
        target_role : str

    OUTPUT:
        action_score      : int  (0-100)
        evaluation_summary: str
        passed            : bool  (score >= 50)

    Confidence impact:
        pass  → confidence += 1
        fail  → confidence -= 1

    Note: Q&A ephemeral — not stored.
    """

    def __init__(self):
        self.name = "ActionAssessmentAgent"

    def _generate_questions(self, action_title: str, target_role: str) -> list[str]:
        system = (
            "You are an expert technical interviewer. "
            "Generate exactly 10 specific questions to evaluate whether a candidate "
            "has mastered the given learning action for their career goal. "
            "Return ONLY a JSON array of 10 question strings."
        )
        user = (
            f"Career Goal   : {target_role}\n"
            f"Learning Action: {action_title}\n\n"
            "Return: [\"question1\", ..., \"question10\"]"
        )
        raw   = call_llm(system, user, max_tokens=700)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        lines = [l.strip().lstrip("0123456789.)- ") for l in raw.splitlines() if l.strip()]
        return lines[:10]

    def _collect_answers(self, questions: list[str]) -> list[dict]:
        qa = []
        print("  Answer each question. Press Enter to submit.\n")
        for i, q in enumerate(questions, 1):
            print(f"  Q{i}: {q}")
            ans = input("  Your answer: ").strip()
            qa.append({"question": q, "answer": ans})
            print()
        return qa  # ephemeral

    def _evaluate(self, action_title: str, target_role: str, qa_pairs: list[dict]) -> dict:
        system = (
            "You are a strict but fair career skills evaluator. "
            "Evaluate the candidate's mastery of the given action based on their answers. "
            "Return ONLY valid JSON."
        )
        qa_text = "\n".join(f"Q: {p['question']}\nA: {p['answer']}" for p in qa_pairs)
        schema  = json.dumps({
            "action_score": "<int 0-100>",
            "evaluation_summary": "<string, 2-3 sentences>"
        }, indent=2)
        user = (
            f"Career Goal   : {target_role}\n"
            f"Assessed Action: {action_title}\n\n"
            f"Q&A:\n{qa_text}\n\n"
            f"Return this JSON:\n{schema}"
        )
        raw = call_llm(system, user, max_tokens=600)
        return extract_json(raw)

    def run(self, input_data: dict) -> dict:
        action_id    = input_data["action_id"]
        action_title = input_data["action_title"]
        target_role  = input_data["target_role"]

        print_section(f"ACTION ASSESSMENT — {action_id}: {action_title}")

        # Step 1: GPT generates 10 questions
        print("  [1/3] Generating mastery questions via GPT...")
        questions = self._generate_questions(action_title, target_role)
        print(f"  ✓ {len(questions)} questions ready.\n")

        # Step 2: User answers (ephemeral)
        qa_pairs = self._collect_answers(questions)

        # Step 3: GPT evaluates
        print("  [3/3] Evaluating mastery via GPT...")
        result = self._evaluate(action_title, target_role, qa_pairs)
        # qa_pairs discarded here

        score  = int(result.get("action_score", 0))
        passed = score >= 50

        print(f"\n  Score : {score}/100  |  {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"  Summary: {result.get('evaluation_summary', '')}")

        return {
            "action_score":       score,
            "evaluation_summary": result.get("evaluation_summary", ""),
            "passed":             passed
        }


# ═══════════════════════════════════════════════════════════════════
#  AGENT 5 — ReroutingAgent
# ═══════════════════════════════════════════════════════════════════

class ReroutingAgent:
    """
    Analyses confidence and roadmap progress to suggest role changes.

    INPUT:
        confidence_score            : int
        current_target_role         : str
        previous_target_role        : str | None
        roadmap_completion_percentage: float

    OUTPUT:
        reroute_suggestion          : bool
        suggested_roles             : list[str]
        return_previous_role_available: bool
        reason                      : str
    """

    def __init__(self):
        self.name = "ReroutingAgent"

    def run(self, input_data: dict) -> dict:
        confidence   = input_data["confidence_score"]
        current_role = input_data["current_target_role"]
        prev_role    = input_data.get("previous_target_role")
        completion   = input_data.get("roadmap_completion_percentage", 0.0)

        reroute_suggestion = False
        suggested_roles    = []
        return_available   = False
        reason             = ""

        # Logic Rule 1: Low confidence → suggest safer roles
        if confidence < 40:
            reroute_suggestion = True
            reason = (
                f"Confidence score ({confidence}) is below the 40-point threshold. "
                "Consider a safer adjacent role to rebuild momentum."
            )
            suggested_roles = self._get_safer_roles(current_role)

        # Logic Rule 2: High confidence or roadmap complete → allow return to previous role
        if confidence >= 80 or completion >= 100.0:
            if prev_role:
                return_available = True
                if not reason:
                    reason = (
                        f"Excellent progress! Confidence ({confidence}) and completion "
                        f"({completion:.1f}%) indicate you may be ready to revisit "
                        f"your previous target: {prev_role}."
                    )

        if not reason:
            reason = f"Confidence ({confidence}) and progress ({completion:.1f}%) are within normal range. Stay on course."

        return {
            "reroute_suggestion":            reroute_suggestion,
            "suggested_roles":               suggested_roles,
            "return_previous_role_available": return_available,
            "reason":                        reason
        }

    def _get_safer_roles(self, current_role: str) -> list[str]:
        """Ask GPT for 3 safer adjacent roles when confidence is low."""
        system = "You are a career advisor. Return ONLY a JSON array of 3 role name strings."
        user   = (
            f"The candidate is struggling with: {current_role}\n"
            "Suggest 3 safer, adjacent roles with lower entry barriers. "
            "Return: [\"role1\", \"role2\", \"role3\"]"
        )
        raw   = call_llm(system, user, max_tokens=200)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return ["Junior " + current_role, "Associate " + current_role, current_role + " Trainee"]


# ═══════════════════════════════════════════════════════════════════
#  AGENT 6 — FeedbackAgent
# ═══════════════════════════════════════════════════════════════════

class FeedbackAgent:
    """
    Generates comprehensive progress feedback.

    INPUT:
        target_role                : str
        confidence_score           : int
        completed_actions_count    : int
        failed_actions_count       : int
        roadmap_progress_percentage: float

    OUTPUT: { feedback_analysis: { ... } } — exact schema as specified.
    """

    def __init__(self):
        self.name = "FeedbackAgent"

    def run(self, input_data: dict) -> dict:
        target_role  = input_data["target_role"]
        confidence   = input_data["confidence_score"]
        completed    = input_data["completed_actions_count"]
        failed       = input_data["failed_actions_count"]
        progress_pct = input_data["roadmap_progress_percentage"]
        next_date    = (date.today() + timedelta(days=30)).isoformat()

        print_section("FEEDBACK ANALYSIS")
        print("  Generating feedback report via GPT...")

        system = (
            "You are an expert AI career coach. "
            "Generate comprehensive progress feedback. "
            "Return ONLY valid JSON matching the exact schema provided."
        )

        schema = {
            "feedback_analysis": {
                "overall_progress_rating": "<Excellent|Good|Fair|Needs Improvement>",
                "progress_percentage": progress_pct,
                "velocity_assessment": "<string: pace of progress>",
                "confidence_adjustment": "<float: suggested adjustment e.g. 0.05>",
                "updated_confidence_score": "<float: new confidence e.g. 0.75>",
                "risk_adjustment": "<string: risk assessment change>",
                "updated_deviation_risk": "<low|medium|high>",
                "strengths_observed": ["<strength1>", "<strength2>"],
                "areas_of_concern": ["<concern1>", "<concern2>"],
                "learning_insights": [
                    {
                        "insight": "<key learning insight>",
                        "evidence": "<what evidence supports this>",
                        "recommendation": "<what to do about it>"
                    }
                ],
                "action_effectiveness": [
                    {
                        "action_id": "action_1",
                        "effectiveness": "<high|medium|low>",
                        "time_efficiency": "<high|medium|low>",
                        "impact_on_goal": "<high|medium|low>",
                        "lessons_learned": ["<lesson1>"]
                    }
                ],
                "motivation_level": "<high|medium|low>",
                "recommended_adjustments": [
                    {
                        "adjustment_type": "<pacing|focus|skills|mindset>",
                        "reason": "<why this adjustment>",
                        "specific_change": "<what exactly to change>"
                    }
                ],
                "next_checkpoint_date": next_date,
                "encouragement_message": "<personalised motivational message>"
            }
        }

        user = (
            f"Career Navigator Progress Report\n"
            f"  Target Role           : {target_role}\n"
            f"  Confidence Score      : {confidence}/100\n"
            f"  Completed Actions     : {completed}\n"
            f"  Failed Actions        : {failed}\n"
            f"  Roadmap Completion    : {progress_pct:.1f}%\n\n"
            f"Fill in this exact JSON schema:\n{json.dumps(schema, indent=2)}"
        )

        raw    = call_llm(system, user, max_tokens=2000)
        result = extract_json(raw)

        # Ensure next_checkpoint_date is always correct
        result["feedback_analysis"]["next_checkpoint_date"] = next_date

        print("  ✓ Feedback report generated.")
        return result


# ═══════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Central controller of the Agentic Career Navigator.

    Responsibilities:
      - Maintains the user_state dict (MongoDB-ready schema)
      - Instantiates and calls agents in the correct order
      - Handles confidence updates after each action assessment
      - Handles role switching (stores previous role, regenerates roadmap + market)
      - Drives the full CLI interaction loop
    """

    # ── State schema (MongoDB-ready) ──────────────────────────────
    INITIAL_STATE = {
        "profile": {
            "target_role":          None,
            "previous_target_role": None,
            "strengths":            [],
            "weaknesses":           [],
            "skills":               []
        },
        "readiness": {
            "score":                  0,
            "status":                 "",
            "evaluation_summary":     "",
            "safer_adjacent_roles":   [],
            "advanced_adjacent_roles":[]
        },
        "confidence_score":    0,
        "market_analysis":     {},
        "roadmap":             {"generated_for_role": "", "steps": []},
        "feedback_analysis":   {},
        "analytics": {
            "completed_actions_count": 0,
            "failed_actions_count":    0
        }
    }

    def __init__(self):
        # Deep-copy the schema template to live state
        self.user_state = json.loads(json.dumps(self.INITIAL_STATE))
        
        # Initialize context manager and MongoDB
        self.context_manager = UserContextManager()
        self.mongo = MongoStore()

        # Instantiate all agents
        self.resume_agent      = ResumeAnalyzerAgent()
        self.readiness_agent   = ReadinessAssessmentAgent()
        self.market_agent      = MarketIntelligenceAgent()
        self.roadmap_agent     = RoadmapAgent()
        self.action_agent      = ActionAssessmentAgent()
        self.rerouting_agent   = ReroutingAgent()
        self.feedback_agent    = FeedbackAgent()

    # ── Helpers ───────────────────────────────────────────────────

    def _get_roadmap_completion(self) -> float:
        """Calculate % of roadmap actions that are not 'pending'."""
        steps = self.user_state["roadmap"].get("steps", [])
        total = sum(len(s.get("actions", [])) for s in steps)
        done  = sum(
            1 for s in steps
            for a in s.get("actions", [])
            if a.get("status") != "pending"
        )
        return (done / total * 100) if total else 0.0

    def _update_confidence(self, delta: int) -> None:
        """Clamp confidence_score to [0, 100]."""
        self.user_state["confidence_score"] = max(
            0, min(100, self.user_state["confidence_score"] + delta)
        )

    def _switch_role(self, new_role: str) -> None:
        """
        Handle target role switch:
          - Store current role as previous
          - Set new role
          - Regenerate roadmap and market analysis
        """
        current = self.user_state["profile"]["target_role"]
        self.user_state["profile"]["previous_target_role"] = current
        self.user_state["profile"]["target_role"]          = new_role

        print(f"\n  ⟳ Switching role: {current} → {new_role}")
        print(f"  Previous role saved: {current}")

        # Regenerate market analysis for new role
        market_result = self.market_agent.run({"target_role": new_role})
        self.user_state["market_analysis"] = market_result.get("market_analysis", {})

        # Regenerate roadmap for new role
        roadmap_result = self.roadmap_agent.run({
            "target_role": new_role,
            "strengths":   self.user_state["profile"]["strengths"],
            "weaknesses":  self.user_state["profile"]["weaknesses"],
            "skills":      self.user_state["profile"]["skills"]
        })
        self.user_state["roadmap"] = roadmap_result

        # Reset analytics
        self.user_state["analytics"]["completed_actions_count"] = 0
        self.user_state["analytics"]["failed_actions_count"]    = 0

        print(f"  ✓ Roadmap and market analysis regenerated for: {new_role}")

    # ── Step 1: Collect user profile ──────────────────────────────

    def _collect_profile(self) -> None:
        print_section("USER PROFILE SETUP")

        target_role = input("  Enter your target career role: ").strip()
        strengths_raw = input(
            "  List your strengths (comma-separated, e.g. communication, Python): "
        ).strip()
        weaknesses_raw = input(
            "  List your weaknesses (comma-separated, e.g. public speaking, SQL): "
        ).strip()
        skills_raw = input(
            "  List your current skills (comma-separated, e.g. Excel, Figma): "
        ).strip()

        self.user_state["profile"]["target_role"] = target_role
        self.user_state["profile"]["strengths"]   = [s.strip() for s in strengths_raw.split(",") if s.strip()]
        self.user_state["profile"]["weaknesses"]  = [w.strip() for w in weaknesses_raw.split(",") if w.strip()]
        self.user_state["profile"]["skills"]      = [s.strip() for s in skills_raw.split(",") if s.strip()]

        print(f"\n  ✓ Profile saved for: {target_role}")

    # ── Step 1.5: Resume Upload (Optional) ──────────────────────────

    def _collect_resume(self, user_id: str) -> None:
        """Optional: Upload and analyze resume"""
        print_section("RESUME UPLOAD (Optional)")
        print("  You can upload a resume (PDF/PNG/JPG) to auto-populate your profile.")
        upload = input("  Upload resume? (y/n): ").strip().lower()
        
        if upload != 'y':
            print("  Skipping resume upload.")
            return
        
        file_path = input("  Enter full path to resume file: ").strip()
        
        if not os.path.exists(file_path):
            print("  ✗ File not found.")
            return
        
        print("\n  Processing resume...")
        result = self.resume_agent.run({
            "user_id": user_id,
            "file_path": file_path,
            "file_name": os.path.basename(file_path)
        })
        
        if result["status"] == "success":
            print("\n  ✓ Resume analysis complete!")
            
            # Auto-populate skills from resume
            skills = result["extracted_skills"]
            extracted_skills = []
            for category, skill_list in skills.items():
                extracted_skills.extend(skill_list)
            
            self.user_state["profile"]["skills"].extend(extracted_skills)
            self.user_state["profile"]["skills"] = list(set(self.user_state["profile"]["skills"]))
            
            print(f"  ✓ {len(extracted_skills)} skills added from resume")
            print(f"  ✓ Skill gap analysis: {result['skill_gap_analysis']['match_percentage']:.1f}% match")
        else:
            print(f"  ✗ Resume analysis failed: {result.get('message')}")

    # ── Step 2: Run Readiness Assessment ──────────────────────────

    def _run_readiness(self) -> None:
        profile = self.user_state["profile"]
        result  = self.readiness_agent.run({
            "target_role": profile["target_role"],
            "skills":      profile["skills"],
            "strengths":   profile["strengths"]
        })

        self.user_state["readiness"] = {
            "score":                   int(result.get("readiness_score", 0)),
            "status":                  result.get("readiness_status", ""),
            "evaluation_summary":      result.get("evaluation_summary", ""),
            "safer_adjacent_roles":    result.get("safer_adjacent_roles", []),
            "advanced_adjacent_roles": result.get("advanced_adjacent_roles", [])
        }

        # Confidence score initialises to readiness_score
        self.user_state["confidence_score"] = self.user_state["readiness"]["score"]

        print_section("READINESS RESULT")
        print_dict(self.user_state["readiness"])
        print(f"\n  Initial Confidence Score: {self.user_state['confidence_score']}")

    # ── Step 3: Run Market Intelligence ───────────────────────────

    def _run_market_intelligence(self) -> None:
        result = self.market_agent.run({
            "target_role": self.user_state["profile"]["target_role"]
        })
        self.user_state["market_analysis"] = result.get("market_analysis", {})
        print_section("MARKET ANALYSIS RESULT")
        print_dict(self.user_state["market_analysis"])

    # ── Step 4: Generate Roadmap ───────────────────────────────────

    def _run_roadmap(self) -> None:
        profile = self.user_state["profile"]
        result  = self.roadmap_agent.run({
            "target_role": profile["target_role"],
            "strengths":   profile["strengths"],
            "weaknesses":  profile["weaknesses"],
            "skills":      profile["skills"]
        })
        self.user_state["roadmap"] = result

        print_section("ROADMAP GENERATED")
        for step in result["steps"]:
            print(f"\n  Month {step['month']}: {step['step_title']}")
            for action in step["actions"]:
                print(f"    [{action['action_id']}] Week {action['week']}: {action['action_title']}")

    # ── Step 5: Action Assessment Loop ────────────────────────────

    def _run_action_loop(self) -> None:
        """
        Interactive loop: user picks actions to assess one at a time.
        After each assessment, check rerouting conditions.
        """
        target_role = self.user_state["profile"]["target_role"]
        steps       = self.user_state["roadmap"]["steps"]

        while True:
            print_section("ACTION ASSESSMENT MENU")
            completion = self._get_roadmap_completion()
            print(f"  Confidence: {self.user_state['confidence_score']}  |  "
                  f"Progress: {completion:.1f}%\n")

            # Build flat list of pending actions for display
            pending = []
            for step in steps:
                for action in step["actions"]:
                    if action["status"] == "pending":
                        pending.append((step, action))

            if not pending:
                print("  🎉 All roadmap actions completed!")
                break

            print("  Pending actions:")
            for i, (step, action) in enumerate(pending, 1):
                print(f"    [{i}] Month {step['month']} | {action['action_id']}: {action['action_title']}")

            print("\n  Options:")
            print("    [number] → assess that action")
            print("    [f]      → generate feedback report")
            print("    [r]      → check rerouting options")
            print("    [q]      → quit and show final state")
            choice = input("\n  Your choice: ").strip().lower()

            if choice == "q":
                break
            elif choice == "f":
                self._run_feedback()
            elif choice == "r":
                self._run_rerouting()
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(pending):
                    step, action = pending[idx]
                    self._assess_action(step, action)
                    # Check rerouting automatically after each assessment
                    self._auto_rerouting_check()
                else:
                    print("  ✗ Invalid selection.")
            else:
                print("  ✗ Unrecognised option.")

    def _assess_action(self, step: dict, action: dict) -> None:
        """Run ActionAssessmentAgent for one action and update state."""
        result = self.action_agent.run({
            "action_id":    action["action_id"],
            "action_title": action["action_title"],
            "target_role":  self.user_state["profile"]["target_role"]
        })

        # Update action in roadmap
        action["score"]  = result["action_score"]
        action["status"] = "passed" if result["passed"] else "failed"

        # Update analytics and confidence
        if result["passed"]:
            self.user_state["analytics"]["completed_actions_count"] += 1
            self._update_confidence(+1)
            print(f"  ✓ Confidence → {self.user_state['confidence_score']} (+1)")
        else:
            self.user_state["analytics"]["failed_actions_count"] += 1
            self._update_confidence(-1)
            print(f"  ✗ Confidence → {self.user_state['confidence_score']} (-1)")

    def _auto_rerouting_check(self) -> None:
        """After each action, silently check if rerouting should be triggered."""
        confidence  = self.user_state["confidence_score"]
        completion  = self._get_roadmap_completion()
        prev_role   = self.user_state["profile"]["previous_target_role"]

        if confidence < 40:
            print(f"\n  ⚠ Confidence dropped to {confidence} — rerouting check triggered.")
            self._run_rerouting()
        elif (confidence >= 80 or completion >= 100.0) and prev_role:
            print(f"\n  🌟 High performance detected — you may return to: {prev_role}")
            choice = input("  Return to previous role? (y/n): ").strip().lower()
            if choice == "y":
                self._switch_role(prev_role)

    # ── Step 6: Rerouting ─────────────────────────────────────────

    def _run_rerouting(self) -> None:
        completion = self._get_roadmap_completion()
        result     = self.rerouting_agent.run({
            "confidence_score":             self.user_state["confidence_score"],
            "current_target_role":          self.user_state["profile"]["target_role"],
            "previous_target_role":         self.user_state["profile"]["previous_target_role"],
            "roadmap_completion_percentage": completion
        })

        print_section("REROUTING ANALYSIS")
        print_dict(result)

        if result["reroute_suggestion"]:
            print("\n  Suggested safer roles:")
            for i, role in enumerate(result["suggested_roles"], 1):
                print(f"    [{i}] {role}")
            print(f"    [0] Stay with: {self.user_state['profile']['target_role']}")

            choice = input("\n  Select a role to switch to (number), or 0 to stay: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(result["suggested_roles"]):
                new_role = result["suggested_roles"][int(choice) - 1]
                self._switch_role(new_role)

        if result["return_previous_role_available"]:
            prev = self.user_state["profile"]["previous_target_role"]
            print(f"\n  You can return to your previous role: {prev}")
            choice = input("  Return to previous role? (y/n): ").strip().lower()
            if choice == "y":
                self._switch_role(prev)

    # ── Step 7: Feedback Report ───────────────────────────────────

    def _run_feedback(self) -> None:
        completion = self._get_roadmap_completion()
        result     = self.feedback_agent.run({
            "target_role":                 self.user_state["profile"]["target_role"],
            "confidence_score":            self.user_state["confidence_score"],
            "completed_actions_count":     self.user_state["analytics"]["completed_actions_count"],
            "failed_actions_count":        self.user_state["analytics"]["failed_actions_count"],
            "roadmap_progress_percentage": completion
        })
        self.user_state["feedback_analysis"] = result.get("feedback_analysis", {})
        print_section("FEEDBACK REPORT")
        print_dict(self.user_state["feedback_analysis"])

    # ── Step 1: User Login/Initialization ────────────────────────

    def _user_login(self) -> tuple[str, Dict[str, Any]]:
        """Initialize or load existing user"""
        print_section("WELCOME TO AGENTIC CAREER NAVIGATOR")
        
        choice = input("  [1] Create new profile\n  [2] Login with email\n  [3] Browse existing users\n  Your choice: ").strip()
        
        if choice == "2":
            return self._load_existing_user_by_email()
        elif choice == "3":
            return self._load_existing_user_by_name()
        else:
            return self._create_new_user()
    
    def _load_existing_user_by_email(self) -> tuple[str, Dict[str, Any]]:
        """Load user by email (primary login method)"""
        email = input("\n  Enter your email: ").strip()
        
        if not email:
            print("  ✗ Email is required. Creating new profile...")
            return self._create_new_user()
        
        user_id = self.context_manager.get_user_id_by_email(email)
        
        if user_id:
            context = self.context_manager.load_context(user_id)
            user_name = context.get('profile', {}).get('name') or 'User'
            print(f"  ✓ Welcome back, {user_name}!")
            return user_id, context
        else:
            print(f"  ✗ No account found for {email}.")
            create = input("  Would you like to create a new account? (y/n): ").strip().lower()
            if create == 'y':
                return self._create_new_user_with_email(email)
            else:
                print("  Returning to main menu...")
                return self._user_login()
    
    def _load_existing_user_by_name(self) -> tuple[str, Dict[str, Any]]:
        """Load user by selecting from existing users by name"""
        users = self.context_manager.list_all_users()
        
        if not users:
            print("  ✗ No existing users found. Creating new profile...")
            return self._create_new_user()
        
        print("\n  Available users:")
        print(f"  {'#':<4} {'Name':<25} {'Email':<30} {'Role':<20}")
        print("  " + "-"*79)
        
        for idx, user in enumerate(users, 1):
            name = (user.get('name') or 'Unknown')[:24]
            email = (user.get('email') or 'No email')[:29]
            role = (user.get('target_role') or 'Not set')[:19]
            print(f"  [{idx}]  {name:<25} {email:<30} {role:<20}")
        
        try:
            choice = int(input(f"\n  Select user (1-{len(users)}): ").strip())
            if 1 <= choice <= len(users):
                selected_user = users[choice - 1]
                user_id = selected_user['user_id']
                context = self.context_manager.load_context(user_id)
                selected_name = selected_user.get('name') or 'User'
                print(f"  ✓ Profile loaded: {selected_name}")
                return user_id, context
            else:
                print("  ✗ Invalid selection. Creating new profile...")
                return self._create_new_user()
        except ValueError:
            print("  ✗ Invalid input. Creating new profile...")
            return self._create_new_user()

    def _create_new_user(self) -> tuple[str, Dict[str, Any]]:
        """Create a new user profile with name and email"""
        print_section("CREATE NEW PROFILE")
        
        # Collect user information
        name = input("  Enter your full name: ").strip()
        email = input("  Enter your email: ").strip()
        
        # Validate email
        if not email or '@' not in email:
            print("  ✗ Valid email is required.")
            return self._create_new_user()
        
        # Check if email already exists
        existing_user_id = self.context_manager.get_user_id_by_email(email)
        if existing_user_id:
            print(f"  ✗ An account with email {email} already exists.")
            load = input("  Would you like to load that account? (y/n): ").strip().lower()
            if load == 'y':
                context = self.context_manager.load_context(existing_user_id)
                return existing_user_id, context
            else:
                return self._create_new_user()
        
        # Create new user
        user_id = f"user_{int(time.time())}"
        context = self.context_manager.initialize_context(user_id)
        
        # Update profile with user info
        context['profile']['name'] = name
        context['profile']['email'] = email
        self.context_manager.save_context(user_id, context)
        
        print(f"\n  ✓ Account created successfully!")
        print(f"  Name: {name}")
        print(f"  Email: {email}")
        print(f"  User ID: {user_id}")
        
        return user_id, context
    
    def _create_new_user_with_email(self, email: str) -> tuple[str, Dict[str, Any]]:
        """Create a new user profile with pre-filled email"""
        print_section("CREATE NEW PROFILE")
        
        name = input("  Enter your full name: ").strip()
        
        if not name:
            name = email.split('@')[0]  # Use email prefix as fallback
        
        # Create new user
        user_id = f"user_{int(time.time())}"
        context = self.context_manager.initialize_context(user_id)
        
        # Update profile with user info
        context['profile']['name'] = name
        context['profile']['email'] = email
        self.context_manager.save_context(user_id, context)
        
        print(f"\n  ✓ Account created successfully!")
        print(f"  Name: {name}")
        print(f"  Email: {email}")
        print(f"  User ID: {user_id}")
        
        return user_id, context

    # ── Step 2: Resume vs Manual Skills Input ─────────────────────

    def _get_initial_skills(self, user_id: str, context: Dict) -> Dict[str, Any]:
        """Option 1: Upload resume OR Option 2: Enter skills manually"""
        print_section("SKILL ENTRY METHOD")
        
        print("  How would you like to provide your skills?")
        choice = input("  [1] Upload resume (PDF/PNG/JPG)\n  [2] Enter skills manually\n  Your choice: ").strip()
        
        if choice == "1":
            return self._handle_resume_upload(user_id, context)
        else:
            return self._handle_manual_skills_entry(context)

    def _handle_resume_upload(self, user_id: str, context: Dict) -> Dict[str, Any]:
        """Upload and analyze resume"""
        print_section("RESUME UPLOAD")
        
        file_path = input("  Enter full path to resume file: ").strip()
        
        if not os.path.exists(file_path):
            print("  ✗ File not found. Falling back to manual entry...")
            return self._handle_manual_skills_entry(context)
        
        # Copy resume file to archive directory
        resume_upload_dir = os.getenv("RESUME_UPLOAD_DIR", "./data/resumes")
        os.makedirs(resume_upload_dir, exist_ok=True)
        
        original_filename = os.path.basename(file_path)
        archived_filename = f"{user_id}_{original_filename}"
        archived_path = os.path.join(resume_upload_dir, archived_filename)
        
        try:
            shutil.copy2(file_path, archived_path)
            print(f"  ✓ Resume archived to: {archived_path}")
        except Exception as e:
            print(f"  ⚠ Archive warning: {str(e)[:60]}")
        
        print("\n  Processing resume...")
        result = self.resume_agent.run({
            "user_id": user_id,
            "file_path": file_path,
            "file_name": os.path.basename(file_path)
        })
        
        if result["status"] == "success":
            print("\n  ✓ Resume analyzed successfully!")
            
            # Extract available information
            parsed = result["parsed_profile"]
            skills_dict = result["extracted_skills"]
            
            # Flatten skills
            extracted_skills = []
            for category, skill_list in skills_dict.items():
                extracted_skills.extend(skill_list)
            
            # Update context with extracted data
            context["profile"].update({
                "name": parsed.get("name"),
                "email": parsed.get("email"),
                "phone": parsed.get("phone"),
                "experience_years": parsed.get("experience_years", 0),
                "resume_uploaded": True,
                "resume_uploaded_at": datetime.now().isoformat(),
                "resume_file_name": original_filename,
                "resume_archived_path": archived_path,
                "skills": {
                    "technical": skills_dict.get("programming_languages", []),
                    "frameworks": skills_dict.get("frameworks", []),
                    "databases": skills_dict.get("databases", []),
                    "tools": skills_dict.get("tools", []) + skills_dict.get("cloud_platforms", []),
                    "soft_skills": skills_dict.get("soft_skills", [])
                }
            })
            
            # Also save resume analysis
            context["resume_analysis"] = {
                "parsed_profile": parsed,
                "extracted_skills": skills_dict,
                "normalized_skills": result["normalized_skills"],
                "upload_time": datetime.now().isoformat(),
                "archived_path": archived_path,
                "original_filename": original_filename
            }
            
            # Persist to database immediately
            self.context_manager.save_context(user_id, context)
            self.mongo.sync(user_id, context)
            
            return {
                "source": "resume",
                "skills": extracted_skills,
                "profile_data": parsed,
                "extracted_skills_dict": skills_dict
            }
        else:
            print(f"  ✗ Resume analysis failed: {result.get('message')}")
            return self._handle_manual_skills_entry(context)

    def _handle_manual_skills_entry(self, context: Dict) -> Dict[str, Any]:
        """Manual skill entry"""
        print_section("MANUAL SKILL ENTRY")
        
        strengths_raw = input("  List your strengths (comma-separated, e.g. communication, Python): ").strip()
        weaknesses_raw = input("  List your weaknesses (comma-separated, e.g. public speaking, SQL): ").strip()
        skills_raw = input("  List your current skills (comma-separated, e.g. Excel, Figma): ").strip()
        
        strengths = [s.strip() for s in strengths_raw.split(",") if s.strip()]
        weaknesses = [w.strip() for w in weaknesses_raw.split(",") if w.strip()]
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
        
        return {
            "source": "manual",
            "skills": skills,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "profile_data": {}
        }

    # ── Step 3: Get Target Role  ──────────────────────────────────

    def _get_target_role(self) -> str:
        """Ask user for their target career role"""
        print_section("TARGET ROLE")
        
        target_role = input("  What is your target career role? ").strip()
        if not target_role:
            target_role = input("  Enter at least one role: ").strip()
        
        print(f"  ✓ Target role set: {target_role}")
        return target_role

    # ── Step 4: Run Readiness Assessment ───────────────────────────

    def _run_readiness_with_skills(self, target_role: str, skills: List[str], strengths: List[str], weaknesses: List[str], user_id: str, context: Dict) -> None:
        """Run readiness assessment with extracted/manual skills"""
        profile = {
            "target_role": target_role,
            "skills": skills,
            "strengths": strengths if strengths else ["To be determined"]
        }
        
        result = self.readiness_agent.run(profile)
        
        # Update user_state with final profile data
        self.user_state["profile"]["target_role"] = target_role
        self.user_state["profile"]["strengths"] = strengths
        self.user_state["profile"]["weaknesses"] = weaknesses if weaknesses else []
        self.user_state["profile"]["skills"] = skills
        
        # Update readiness results
        self.user_state["readiness"] = {
            "score": int(result.get("readiness_score", 0)),
            "status": result.get("readiness_status", ""),
            "evaluation_summary": result.get("evaluation_summary", ""),
            "safer_adjacent_roles": result.get("safer_adjacent_roles", []),
            "advanced_adjacent_roles": result.get("advanced_adjacent_roles", [])
        }
        
        # Initialize confidence score from readiness
        self.user_state["confidence_score"] = self.user_state["readiness"]["score"]
        
        # Update context
        context["career_state"]["current_target_role"] = target_role
        context["readiness_assessment"].update({
            "status": result.get("readiness_status"),
            "confidence_score": self.user_state["readiness"]["score"],
            "reasoning": result.get("evaluation_summary"),
            "last_assessed_at": datetime.now().isoformat()
        })
        
        # Persist
        self.context_manager.save_context(user_id, context)
        self.mongo.sync(user_id, context)
        
        print_section("READINESS RESULT")
        print_dict(self.user_state["readiness"])
        print(f"\n  Initial Confidence Score: {self.user_state['confidence_score']}")

    # ── Main run method ───────────────────────────────────────────

    def _has_existing_progress(self, context: Dict[str, Any]) -> bool:
        """Check if user has existing progress/roadmap"""
        has_target_role = context.get("career_state", {}).get("current_target_role") is not None
        
        # Check both possible roadmap structures
        roadmap = context.get("active_roadmap", {})
        has_phases = len(roadmap.get("phases", [])) > 0
        has_steps = len(roadmap.get("steps", [])) > 0
        has_generated_roadmap = roadmap.get("generated_for_role") is not None
        
        has_roadmap = has_phases or has_steps or has_generated_roadmap
        
        return has_target_role and has_roadmap
    
    def _show_returning_user_menu(self, user_id: str, context: Dict[str, Any]) -> str:
        """Show menu for returning users"""
        print_section("WELCOME BACK!")
        
        name = context.get("profile", {}).get("name", "User")
        target_role = context.get("career_state", {}).get("current_target_role")
        progress = context.get("active_roadmap", {}).get("completion_percentage", 0)
        confidence = context.get("readiness_assessment", {}).get("confidence_score", 0)
        
        print(f"  Name: {name}")
        print(f"  Current Goal: {target_role}")
        print(f"  Progress: {progress:.0f}%")
        print(f"  Confidence: {confidence:.0%}")
        
        print("\n  What would you like to do?")
        print("  [1] Continue my current roadmap")
        print("  [2] Start fresh with a new goal")
        print("  [3] View my progress report")
        
        choice = input("\n  Your choice: ").strip()
        return choice
    
    def _load_existing_roadmap(self, user_id: str, context: Dict[str, Any]) -> None:
        """Load existing user roadmap into user_state"""
        # Load profile data
        self.user_state["profile"]["target_role"] = context.get("career_state", {}).get("current_target_role")
        self.user_state["profile"]["skills"] = context.get("profile", {}).get("skills", [])
        self.user_state["profile"]["strengths"] = context.get("profile", {}).get("strengths", [])
        self.user_state["profile"]["weaknesses"] = context.get("profile", {}).get("weaknesses", [])
        
        # Load confidence and readiness
        self.user_state["confidence_score"] = context.get("readiness_assessment", {}).get("confidence_score", 0)
        self.user_state["readiness"] = context.get("readiness_assessment", {})
        
        # Load roadmap (handle both phases and steps structures)
        active_roadmap = context.get("active_roadmap", {})
        self.user_state["roadmap"] = active_roadmap
        
        # Load analytics
        self.user_state["analytics"]["completed_actions_count"] = context.get("progress", {}).get("actions_completed", 0)
        self.user_state["analytics"]["failed_actions_count"] = context.get("progress", {}).get("actions_failed", 0)
        
        # Determine total actions for progress calculation  
        total_actions = 0
        if "steps" in active_roadmap:
            for step in active_roadmap.get("steps", []):
                total_actions += len(step.get("actions", []))
        elif "phases" in active_roadmap:
            for phase in active_roadmap.get("phases", []):
                total_actions += len(phase.get("actions", []))
        
        print(f"\n  ✓ Loaded existing roadmap for: {self.user_state['profile']['target_role']}")
        print(f"  Actions completed: {self.user_state['analytics']['completed_actions_count']}/{total_actions}")
        print(f"  Current confidence: {self.user_state['confidence_score']:.0%}")
    
    def _show_progress_summary(self, context: Dict[str, Any]) -> None:
        """Display current progress summary"""
        print_section("YOUR PROGRESS SUMMARY")
        
        name = context.get("profile", {}).get("name", "User")
        target_role = context.get("career_state", {}).get("current_target_role", "Not set")
        confidence = context.get("readiness_assessment", {}).get("confidence_score", 0)
        progress = context.get("active_roadmap", {}).get("completion_percentage", 0)
        actions_completed = context.get("progress", {}).get("actions_completed", 0)
        actions_failed = context.get("progress", {}).get("actions_failed", 0)
        weeks_completed = context.get("progress", {}).get("weeks_completed", 0)
        
        print(f"\n  👤 User: {name}")
        print(f"  🎯 Target Role: {target_role}")
        print(f"  📊 Confidence Score: {confidence:.0%}")
        print(f"  📈 Overall Progress: {progress:.0f}%")
        print(f"  ✅ Actions Completed: {actions_completed}")
        print(f"  ❌ Actions Failed: {actions_failed}")
        print(f"  📅 Weeks Completed: {weeks_completed}")
        
        # Show current week's actions
        current_actions = context.get("current_actions", {}).get("this_week", [])
        if current_actions:
            print(f"\n  📋 This Week's Actions:")
            for action in current_actions:
                status = action.get("status", "pending")
                print(f"     • {action.get('action_title')} [{status}]")
        
        print("\n")
    
    def _persist_final_state(self, user_id: str, context: Dict[str, Any]) -> None:
        """Save final state to database"""
        print_section("FINAL PERSISTENT STATE")
        print_dict(self.user_state)
        
        # Extract current profile data from user_state
        target_role = self.user_state["profile"]["target_role"]
        skills = self.user_state["profile"]["skills"]
        strengths = self.user_state["profile"]["strengths"]
        weaknesses = self.user_state["profile"]["weaknesses"]
        
        # Prepare roadmap data with proper status
        roadmap_data = self.user_state.get("roadmap", {})
        if roadmap_data and roadmap_data.get("steps"):
            roadmap_data["status"] = "in_progress" if self.user_state["analytics"]["completed_actions_count"] > 0 else "generated"
        
        # Persist final state
        try:
            print("\n  Saving to database...")
            context.update({
                "profile": {
                    "target_role": target_role,
                    "skills": skills,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "name": context.get("profile", {}).get("name"),
                    "email": context.get("profile", {}).get("email"),
                    "phone": context.get("profile", {}).get("phone"),
                    "experience_years": context.get("profile", {}).get("experience_years", 0),
                    "resume_uploaded": context.get("profile", {}).get("resume_uploaded", False),
                    "resume_uploaded_at": context.get("profile", {}).get("resume_uploaded_at"),
                    "resume_file_name": context.get("profile", {}).get("resume_file_name")
                },
                "career_state": {
                    "current_target_role": target_role,
                    "role_history": context.get("career_state", {}).get("role_history", [])
                },
                "readiness_assessment": self.user_state.get("readiness", {}),
                "active_roadmap": roadmap_data,
                "progress": {
                    "actions_completed": self.user_state["analytics"]["completed_actions_count"],
                    "actions_failed": self.user_state["analytics"]["failed_actions_count"],
                    "weeks_completed": context.get("progress", {}).get("weeks_completed", 0),
                    "last_activity_at": datetime.now().isoformat()
                }
            })
            self.context_manager.save_context(user_id, context)
            self.mongo.sync(user_id, context)
            print(f"  ✓ Profile saved to database (User ID: {user_id})")
        except Exception as e:
            print(f"  ⚠ Database save error: {str(e)[:100]}")
        
        print("\n  ✓ Session complete!\n")

    def run(self) -> None:
        """
        Complete flow:
          1. User Login/Create Account
          2. Check if returning user with progress
          3. Resume Upload OR Manual Skills Entry (for new/fresh start)
          4. Get Target Role
          5. Run Readiness Assessment
          6. Market Intelligence
          7. Roadmap Generation
          8. Action Assessment Loop
          9. Final Feedback
          10. Persist everything to MongoDB
        """
        print("\n" + "╔" + "═"*58 + "╗")
        print("║        AGENTIC CAREER NAVIGATOR                        ║")
        print("║        Powered by Groq (openai/gpt-oss-120b)           ║")
        print("╚" + "═"*58 + "╝")

        # Step 1: User Login
        user_id, context = self._user_login()
        
        # Step 2: Check if returning user with existing progress
        if self._has_existing_progress(context):
            choice = self._show_returning_user_menu(user_id, context)
            
            if choice == "1":
                # Load existing data and continue
                self._load_existing_roadmap(user_id, context)
                self._run_action_loop()
                self._run_feedback()
                self._persist_final_state(user_id, context)
                return
            elif choice == "3":
                # Show progress and exit
                self._show_progress_summary(context)
                return
            # else choice == "2" or invalid: continue with fresh start below
        
        # Step 3: Get Skills (Resume or Manual) - for new users or fresh start
        skills_input = self._get_initial_skills(user_id, context)
        skills = skills_input.get("skills", [])
        strengths = skills_input.get("strengths", [])
        weaknesses = skills_input.get("weaknesses", [])
        
        print(f"\n  ✓ Skills loaded: {len(skills)} skills extracted")
        
        # Step 3: Get Target Role
        target_role = self._get_target_role()
        
        # Step 4: Readiness Assessment
        self._run_readiness_with_skills(target_role, skills, strengths, weaknesses, user_id, context)
        
        # Step 5: Market Intelligence
        self._run_market_intelligence()
        
        # Step 6: Roadmap
        self.user_state["profile"]["target_role"] = target_role
        self.user_state["profile"]["skills"] = skills
        self.user_state["profile"]["strengths"] = strengths
        self.user_state["profile"]["weaknesses"] = weaknesses
        self._run_roadmap()
        
        # Step 7: Action Loop
        self._run_action_loop()
        
        # Step 8: Final Feedback
        self._run_feedback()
        
        # Step 9: Final save to database
        self._persist_final_state(user_id, context)


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS (View Progress / User Management)
# ═══════════════════════════════════════════════════════════════════

def view_all_users() -> None:
    """Display all existing users with their progress"""
    manager = UserContextManager()
    users = manager.list_all_users()
    
    if not users:
        print("❌ No existing users found.\n")
        return
    
    print("\n" + "="*120)
    print("📊 ALL USERS - PROGRESS OVERVIEW".center(120))
    print("="*120)
    print(f"{'#':<3} {'Name':<25} {'Email':<30} {'Target Role':<20} {'Progress':<12} {'Weeks':<8}")
    print("-"*120)
    
    for idx, user in enumerate(users, 1):
        name = (user.get('name') or 'Unknown')[:24]
        email = (user.get('email') or 'No email')[:29]
        role = (user.get('target_role') or 'Not set')[:19]
        progress = f"{user.get('progress_percentage', 0):.0f}%"
        weeks = user.get('weeks_completed', 0)
        print(f"{idx:<3} {name:<25} {email:<30} {role:<20} {progress:<12} {weeks:<8}")
    
    print("="*120 + "\n")


def view_user_progress(name: str) -> None:
    """Display detailed progress for a specific user"""
    manager = UserContextManager()
    progress = manager.get_user_progress_by_name(name)
    
    if not progress:
        print(f"\n❌ User '{name}' not found.")
        view_all_users()
        return
    
    display_name = (progress['name'] or 'Unknown').upper()
    print("\n" + "="*80)
    print(f"📈 PROGRESS FOR {display_name}".center(80))
    print("="*80)
    
    print(f"\n🎯 Career Target: {progress['target_role'] or 'Not set'}")
    print(f"   User ID: {progress['user_id']}")
    
    print(f"\n📊 Readiness Assessment:")
    print(f"   Confidence Score: {progress['confidence_score']:.1%}")
    print(f"   Skill Match: {progress['skill_match_percentage']}%")
    
    print(f"\n🗺️  Roadmap Status:")
    print(f"   Status: {progress['roadmap_status'] or 'Not started'}")
    print(f"   Overall Progress: {progress['completion_percentage']:.0f}%")
    print(f"   Current Week: {progress['current_week'] or 'N/A'}")
    print(f"   Weeks Completed: {progress['weeks_completed']}")
    
    print(f"\n✅ Actions Tracking:")
    print(f"   Completed: {progress['actions_completed']}")
    print(f"   Failed: {progress['actions_failed']}")
    print(f"   Total Hours Invested: {progress['total_hours_invested']:.1f}h")
    
    print(f"\n🕐 Last Activity: {progress['last_activity'] or 'Never'}")
    print("="*80 + "\n")


def view_user_menu() -> None:
    """Interactive menu to view user progress"""
    while True:
        print("\n" + "="*80)
        print("📋 USER MANAGEMENT & PROGRESS VIEWER".center(80))
        print("="*80)
        print("\n1. List all users")
        print("2. View progress by name")
        print("3. List users & enter full agent")
        print("4. Exit")
        print("\n" + "="*80)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            view_all_users()
        elif choice == "2":
            name = input("\nEnter user name (or partial name): ").strip()
            if name:
                view_user_progress(name)
            else:
                print("❌ Please enter a name.")
        elif choice == "3":
            orchestrator = Orchestrator()
            orchestrator.run()
        elif choice == "4":
            print("\n👋 Goodbye!\n")
            break
        else:
            print("❌ Invalid option. Please try again.")

# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point with menu to choose between:
    - Run full agent session
    - View user progress
    - Exit
    """
    while True:
        print("\n" + "="*80)
        print("🚀 AGENTIC CAREER NAVIGATOR - MAIN MENU".center(80))
        print("="*80)
        print("\n1. Start/Continue Career Session")
        print("2. View User Progress & Management")
        print("3. Exit")
        print("\n" + "="*80)
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            orchestrator = Orchestrator()
            orchestrator.run()
        elif choice == "2":
            view_user_menu()
        elif choice == "3":
            print("\n👋 Thank you for using Agentic Career Navigator!\n")
            break
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
