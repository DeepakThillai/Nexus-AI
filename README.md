# 🎯 NEXUS-AI: Agentic Career Navigator

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Components](#architecture--components)
3. [Database Schema](#database-schema)
4. [All Agents Explained](#all-agents-explained)
5. [New User Workflow](#new-user-workflow)
6. [Existing User Workflow](#existing-user-workflow)
7. [Rerouting Logic](#rerouting-logic)
8. [File Structure](#file-structure)
9. [Installation & Setup](#installation--setup)

---

## 🏗️ System Overview

**NEXUS-AI** is an intelligent career navigation system powered by multiple specialized AI agents. It guides users through a personalized 5-month learning journey with:

- ✅ **Resume Analysis** or Manual Skill Entry
- ✅ **Readiness Assessment** (personalized Q&A evaluation)
- ✅ **Market Intelligence** (job market analysis)
- ✅ **5-Month Structured Roadmap** (20 actionable steps)
- ✅ **Progress Tracking** with Confidence Scoring
- ✅ **Intelligent Rerouting** (alternative career path suggestions)
- ✅ **Persistent State Management** (JSON + MongoDB)

### Key Features
- **Multi-Agent Architecture**: 7 specialized AI agents coordinate seamlessly
- **Database-First Design**: All data stored in MongoDB with JSON fallback
- **Resume Processing**: Extracts skills from PDF/PNG/JPG files
- **Progressive Learning**: 5 months × 4 actions per month = 20 total steps
- **Confidence Scoring**: Real-time confidence tracking (0-100)
- **Rerouting System**: Suggests alternative roles based on performance
- **Session Continuity**: Resume existing progress where you left off

---

## 🏛️ Architecture & Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│              (Master Coordinator & State)                    │
└┬────────────────────────────────────────────────────────────┘
 │
 ├─► 📄 ResumeAnalyzerAgent      (Resume Processing)
 ├─► 📊 ReadinessAssessmentAgent (Skill Evaluation)
 ├─► 📈 MarketIntelligenceAgent  (Job Market Analysis)
 ├─► 🗺️  RoadmapAgent            (Learning Path Generation)
 ├─► ✅ ActionAssessmentAgent    (Progress Evaluation)
 ├─► 🔄 ReroutingAgent           (Alternative Path Suggestion)
 └─► 📝 FeedbackAgent            (Progress Report)

         ↓ ALL DATA FLOWS ↓

┌─────────────────────────────────────────────────────────────┐
│         DATABASE LAYER                                       │
├─────────────────────────────────────────────────────────────┤
│ JSON Storage: ./data/user_contexts/user_{id}.json           │
│ Resume Archive: ./data/resumes/user_{id}_*.pdf              │
│ MongoDB: career_navigation.user_contexts (Cloud)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 💾 Database Schema

### User Context Structure

```json
{
  "user_id": "user_1771323599",
  "created_at": "2026-02-17T10:30:00",
  "last_updated": "2026-02-17T12:45:00",

  "profile": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "experience_years": 3,
    "education": {
      "degree": "B.Tech Computer Science",
      "institution": "MIT",
      "graduation_year": 2021,
      "cgpa": 3.8
    },
    "skills": {
      "technical": ["Python", "JavaScript", "SQL"],
      "frameworks": ["React", "Django"],
      "databases": ["PostgreSQL", "MongoDB"],
      "tools": ["Git", "Docker", "AWS"],
      "soft_skills": ["Leadership", "Communication"]
    },
    "projects": [...],
    "certifications": [...],
    "resume_uploaded": true,
    "resume_file_name": "sample_resume.pdf"
  },

  "career_state": {
    "current_target_role": "Senior Backend Engineer",
    "original_target_role": "Senior Backend Engineer",
    "role_tier": "advanced",
    "role_history": ["Junior Developer", "Mid-Level Engineer"]
  },

  "readiness_assessment": {
    "status": "qualified",
    "confidence_score": 75,
    "skill_match_percentage": 85,
    "matched_skills": ["Python", "SQL"],
    "missing_skills": ["Kubernetes", "Microservices"],
    "reasoning": "Strong fundamentals, needs DevOps knowledge"
  },

  "active_roadmap": {
    "roadmap_id": "roadmap_123",
    "created_at": "2026-02-17T10:35:00",
    "duration_months": 5,
    "total_weeks": 20,
    "current_phase": 1,
    "current_week": 2,
    "completion_percentage": 10,
    "phases": [
      {
        "month": 1,
        "step_title": "Foundations & Core Concepts",
        "actions": [
          {
            "action_id": "action_1",
            "week": 1,
            "action_title": "Setup Dev Environment",
            "status": "completed",
            "score": 95
          },
          {
            "action_id": "action_2",
            "week": 2,
            "action_title": "Learn Docker Basics",
            "status": "in_progress",
            "score": null
          }
        ]
      }
    ],
    "status": "in_progress"
  },

  "progress": {
    "weeks_completed": 1,
    "actions_completed": 1,
    "actions_failed": 0,
    "current_streak_weeks": 1,
    "total_hours_invested": 8,
    "last_activity_at": "2026-02-17T12:00:00"
  },

  "reroute_state": {
    "is_active": false,
    "reroute_count": 0,
    "can_return_to_previous": true
  },

  "resume_analysis": {
    "parsed_profile": { ... },
    "extracted_skills": { ... },
    "normalized_skills": [...]
  }
}
```

### Storage Locations

| Location | Content | Purpose | Format |
|----------|---------|---------|--------|
| `./data/user_contexts/{user_id}.json` | Complete user context | Local backup & offline access | JSON |
| `./data/resumes/{user_id}_{filename}` | Original uploaded resume | Archive & audit trail | PDF/PNG/JPG |
| `MongoDB: career_navigation.user_contexts` | Complete user context | Cloud sync & queries | BSON |

---

## 🤖 All Agents Explained

### 1. **ResumeAnalyzerAgent**

**Purpose**: Extract structured information from resume files and parse user profile.

**Inputs**:
```json
{
  "user_id": "user_1771323599",
  "file_path": "./sample_resume.pdf",
  "file_name": "sample_resume.pdf"
}
```

**Process**:
1. **Extract Text**: Uses pdfminer.six for PDF or pytesseract for OCR on images
2. **Parse Structure**: Sends extracted text to LLM to structure as JSON
3. **Normalize Skills**: Categorizes skills into technical, frameworks, databases, tools, soft skills
4. **Archive File**: Copies resume to `./data/resumes/user_{id}_{filename}`

**Outputs**:
```json
{
  "status": "success",
  "parsed_profile": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "experience_years": 3,
    "education": "B.Tech Computer Science",
    "linkedin": "linkedin.com/in/johndoe",
    "github": "github.com/johndoe"
  },
  "extracted_skills": {
    "programming_languages": ["Python", "Java", "JavaScript"],
    "frameworks": ["Django", "React"],
    "databases": ["PostgreSQL", "MongoDB"],
    "tools": ["Git", "Docker"],
    "cloud_platforms": ["AWS"],
    "soft_skills": ["Leadership", "Communication"]
  },
  "normalized_skills": ["Python", "Django", "AWS", ...]
}
```

**Key Methods**:
- `extract_text_from_pdf()` → Extracts text from PDF files
- `extract_text_from_image()` → Extracts text via OCR from images
- `parse_resume()` → Uses LLM to structure resume text as JSON
- `normalize_skills()` → Categorizes and deduplicates skills

---

### 2. **ReadinessAssessmentAgent**

**Purpose**: Evaluate user's readiness for target role through interactive Q&A.

**Inputs**:
```json
{
  "target_role": "Senior Backend Engineer",
  "skills": ["Python", "Docker", "PostgreSQL"],
  "strengths": ["Problem solving", "Leadership"]
}
```

**Process**:
1. **Generate Questions**: LLM creates 10 role-specific assessment questions
2. **Collect Answers**: User answers each question via CLI (Q&A stays in memory, not saved)
3. **Evaluate**: LLM scores answers and determines readiness level
4. **Suggest Alternative Roles**: Recommends safer and advanced alternative paths

**Outputs**:
```json
{
  "readiness_score": 75,
  "readiness_status": "qualified",
  "evaluation_summary": "Strong technical foundation but needs DevOps expertise...",
  "safer_adjacent_roles": ["Integration Engineer", "DevOps Engineer"],
  "advanced_adjacent_roles": ["Solutions Architect", "Tech Lead"]
}
```

**Key Methods**:
- `_generate_questions()` → Generates 10 assessment questions via LLM
- `_collect_answers()` → Prompts user for answers (ephemeral)
- `_evaluate()` → Scores answers and generates readiness assessment

**Important Note**: Q&A pairs are NOT stored in persistent storage—only scores and summaries are saved.

---

### 3. **MarketIntelligenceAgent**

**Purpose**: Analyze job market trends and requirements for target role.

**Inputs**:
```json
{
  "target_role": "Senior Backend Engineer"
}
```

**Process**:
1. Research market demand for role
2. Identify trending skills and tools
3. Analyze salary ranges and growth opportunities
4. Generate market insights

**Outputs**:
```json
{
  "market_analysis": {
    "role": "Senior Backend Engineer",
    "market_demand": "high",
    "trending_skills": ["Kubernetes", "Microservices", "gRPC"],
    "market_salary_range": "$100,000 - $150,000",
    "growth_opportunities": "Strong in tech sector",
    "job_market_insights": "..."
  }
}
```

---

### 4. **RoadmapAgent**

**Purpose**: Generate a structured 5-month learning roadmap with 20 actionable steps.

**Inputs**:
```json
{
  "target_role": "Senior Backend Engineer",
  "strengths": ["Problem solving", "Leadership"],
  "weaknesses": ["DevOps", "Cloud infrastructure"],
  "skills": ["Python", "Docker"]
}
```

**Structure**: Exactly 5 months × 4 actions per month = **20 total steps**

**Process**:
1. Analyze target role requirements
2. Consider user's strengths/weaknesses
3. Generate 5 monthly phases, each with 4 weekly actions
4. Each action is progressively challenging

**Outputs**:
```json
{
  "generated_for_role": "Senior Backend Engineer",
  "steps": [
    {
      "month": 1,
      "step_title": "Foundations & Environment Setup",
      "actions": [
        {
          "action_id": "action_1",
          "week": 1,
          "action_title": "Setup Development Environment",
          "status": "pending",
          "score": null
        },
        {
          "action_id": "action_2",
          "week": 2,
          "action_title": "Learn Docker Containerization",
          "status": "pending",
          "score": null
        },
        {
          "action_id": "action_3",
          "week": 3,
          "action_title": "Build Docker-based Project",
          "status": "pending",
          "score": null
        },
        {
          "action_id": "action_4",
          "week": 4,
          "action_title": "Implement Database Optimization",
          "status": "pending",
          "score": null
        }
      ]
    },
    "... repeat months 2-5 with actions 5-20 ..."
  ]
}
```

**Roadmap Structure**:
```
Month 1: Foundations (4 actions)
  Week 1: Setup & Environment
  Week 2: Core Concepts
  Week 3: Practical Application
  Week 4: Mini Project

Month 2: Intermediate Skills (4 actions)
  Week 5: Advanced Topics
  Week 6: Integration
  Week 7: Performance
  Week 8: Deployment

Month 3: Advanced Topics (4 actions)
  Week 9-12: Deeper expertise

Month 4: Professional Skills (4 actions)
  Week 13-16: Architecture & design

Month 5: Mastery & Leadership (4 actions)
  Week 17-20: Career advancement
```

---

### 5. **ActionAssessmentAgent**

**Purpose**: Evaluate mastery level for individual roadmap actions.

**Inputs**:
```json
{
  "action_id": "action_1",
  "action_title": "Setup Development Environment",
  "target_role": "Senior Backend Engineer"
}
```

**Process**:
1. Ask user about their approach to the action
2. Evaluate based on best practices
3. Assign mastery score (0-100)
4. Provide feedback

**Outputs**:
```json
{
  "action_id": "action_1",
  "action_title": "Setup Development Environment",
  "passed": true,
  "action_score": 92,
  "feedback": "Excellent setup with proper tooling...",
  "suggested_improvements": [...]
}
```

**Confidence Impact**:
- ✅ **Passed** → Confidence +1
- ❌ **Failed** → Confidence -1

---

### 6. **ReroutingAgent**

**Purpose**: Suggest alternative career paths when confidence drops or peaks.

**Triggers**:
- ⚠️ Confidence < 40 → Suggest safer roles
- 🌟 Confidence ≥ 80 → Suggest advanced roles or return option

**Inputs**:
```json
{
  "confidence_score": 35,
  "current_target_role": "Senior Backend Engineer",
  "previous_target_role": "Mid-Level Engineer",
  "roadmap_completion_percentage": 25
}
```

**Process**:
1. Analyze confidence trend
2. Identify role misalignment factors
3. Suggest alternative paths
4. Allow user to switch or stay

**Outputs**:
```json
{
  "reroute_suggestion": true,
  "suggested_roles": ["Integration Engineer", "DevOps Engineer", "QA Lead"],
  "return_previous_role_available": true,
  "reroute_reason": "Current path challenging; try more foundational role"
}
```

---

### 7. **FeedbackAgent**

**Purpose**: Generate comprehensive progress reports and insights.

**Inputs**:
```json
{
  "user_id": "user_1771323599",
  "weeks_completed": 8,
  "actions_completed": 8,
  "current_confidence": 72,
  "target_role": "Senior Backend Engineer"
}
```

**Process**:
1. Analyze progress metrics
2. Identify strengths and gaps
3. Generate personalized recommendations
4. Motivate for next phase

**Outputs**:
```json
{
  "progress_summary": "You've completed 40% of your roadmap...",
  "strengths_identified": ["Consistent commitment", "Technical growth"],
  "areas_for_improvement": ["Speed of learning", "Practical application"],
  "next_phase_recommendations": [...],
  "motivational_message": "..."
}
```

---

## 👤 New User Workflow

### Step-by-Step Flow

```
START
  ↓
[1] CREATE ACCOUNT
    └─ Generate unique user_id (timestamp-based)
    └─ Initialize empty user context in MongoDB + JSON
  ↓
[2] SKILL ENTRY CHOICE
    ├─→ Option A: UPLOAD RESUME
    │   ├─ Accept PDF/PNG/JPG
    │ ├─ Archive to ./data/resumes/
    │   ├─ ResumeAnalyzerAgent extracts:
    │   │  ├─ Name, Email, Phone
    │   │  ├─ Experience years
    │   │  └─ Skills (categorized)
    │   └─ Save to context["profile"]["skills"]
    │
    └─→ Option B: MANUAL ENTRY
        ├─ Prompt for strengths (comma-separated)
        ├─ Prompt for weaknesses (comma-separated)
        ├─ Prompt for current skills (comma-separated)
        └─ Save directly to context
  ↓
[3] Ask for TARGET ROLE
    └─ User specifies desired role
    └─ Save to context["career_state"]["current_target_role"]
  ↓
[4] READINESS ASSESSMENT
    └─ ReadinessAssessmentAgent runs:
       ├─ Generate 10 assessment questions
       ├─ Collect user answers (Q&A in memory only)
       ├─ Evaluate via LLM → readiness score
       └─ Suggest alternative roles
    └─ Save score to context["readiness_assessment"]
  ↓
[5] MARKET ANALYSIS
    └─ MarketIntelligenceAgent analyzes:
       ├─ Job market demand
       ├─ Trending skills
       ├─ Salary ranges
       └─ Growth opportunities
    └─ Save to context["readiness_assessment"]["market_analysis"]
  ↓
[6] ROADMAP GENERATION
    └─ RoadmapAgent creates 5-month plan:
       ├─ Exactly 5 months (phases)
       ├─ 4 actions per month (20 total)
       ├─ Progressive difficulty
       └─ Clear weekly milestones
    └─ Save to context["active_roadmap"]["phases"]
  ↓
[7] FIRST ACTION SUGGESTION
    └─ Display Month 1, Week 1 action
    └─ Prompt to start action loop
  ↓
[8] ACTION LOOP (Monthly)
    For each action:
    ├─ Display action title and details
    ├─ ActionAssessmentAgent evaluates mastery
    ├─ User gets score + feedback
    ├─ Confidence +1 (pass) or -1 (fail)
    ├─ Auto-check for rerouting
    └─ Move to next action
  ↓
[9] REROUTING CHECK (Automatic)
    ├─ IF confidence < 40
    │  └─ ReroutingAgent suggests safer roles
    │  └─ User can switch or continue
    │
    └─ IF confidence ≥ 80
       └─ Suggest advanced roles
       └─ Option to return to previous role
  ↓
[10] FINAL FEEDBACK
     └─ FeedbackAgent generates:
        ├─ Progress summary
        ├─ Strengths & weaknesses
        ├─ Recommendations
        └─ Motivational message
  ↓
[11] PERSIST TO DATABASE
     └─ Save full context to:
        ├─ ./data/user_contexts/{user_id}.json
        └─ MongoDB cloud
  ↓
END
```

### Data Flow for New User

```
User Inputs
    ↓
Resume/Manual Skills
    ↓ ResumeAnalyzerAgent (if resume)
    ↓
Context["profile"]["skills"]
    ↓ Target Role Selection
    ↓
Context["career_state"]["current_target_role"]
    ↓ ReadinessAssessmentAgent
    ↓
Context["readiness_assessment"]
    ↓ MarketIntelligenceAgent
    ↓
Context["readiness_assessment"]["market_analysis"]
    ↓ RoadmapAgent
    ↓
Context["active_roadmap"]["phases"]
    ↓ ActionLoop (ActionAssessmentAgent × 20)
    ↓
Context["progress"]
Context["reroute_state"]
    ↓ FeedbackAgent
    ↓
MongoDB + JSON Save
    ↓
Output: Complete Career Plan with Progress Tracking
```

---

## 👥 Existing User Workflow

### Resume Previous Progress

```
START
  ↓
[1] LOGIN / ACCOUNT SELECTION
    ├─ Option A: Create NEW profile → (follow New User Workflow)
    └─ Option B: Load EXISTING profile
       └─ UserContextManager.load_context(user_id)
       └─ Retrieve from MongoDB or JSON
  ↓
[2] DISPLAY RECOVERY STATUS
    Display:
    ├─ Current target role
    ├─ Confidence score
    ├─ Weeks completed
    ├─ Current roadmap progress (e.g., "Month 2, Week 6 of 20")
    └─ Last activity timestamp
  ↓
[3] OPTIONS
    ├─ [A] Continue current roadmap
    │    └─ Load next pending action
    │    └─ Continue action loop
    │
    ├─ [B] Change target role → Regenerate roadmap
    │    └─ Clear old roadmap
    │    └─ Trigger RoadmapAgent with new role
    │
    └─ [C] View progress report
         └─ FeedbackAgent generates summary
  ↓
[4] CONTINUE ACTION LOOP
    From where left off:
    ├─ Retrieve context["active_roadmap"]["current_week"]
    ├─ Load next pending action
    ├─ ActionAssessmentAgent evaluates
    ├─ Update confidence score
    ├─ Auto-trigger rerouting if needed
    └─ Save progress to context["progress"]
  ↓
[5] CONFIDENCE TRACKING
    Real-time updates:
    ├─ Each passed action → +1
    ├─ Each failed action → -1
    ├─ Current score = context["readiness_assessment"]["confidence_score"]
    └─ Saved after each action
  ↓
[6] REROUTING CHECK (Always Active)
    While action loop running:
    ├─ After each action, check:
    │  ├─ IF confidence < 40 AND not already rerouted
    │  │  └─ Suggest safer alternative roles
    │  │
    │  └─ IF confidence ≥ 80
    │     └─ Suggest advanced roles
    │     └─ Option to return to previous role
    │
    └─ Update context["reroute_state"]
  ↓
[7] FINAL SAVE
    └─ Update context with:
       ├─ Progress metrics
       ├─ Latest confidence score
       ├─ Completed actions
       ├─ Current week status
       └─ Timestamps
    └─ Persist to MongoDB + JSON
  ↓
END
```

### Context Recovery Details

When loading existing user:

```python
# Load from database
context = UserContextManager().load_context(user_id)

# Extract current state
current_role = context["career_state"]["current_target_role"]
confidence = context["readiness_assessment"]["confidence_score"]
weeks_done = context["progress"]["weeks_completed"]
current_week = context["active_roadmap"]["current_week"]
roadmap = context["active_roadmap"]["phases"]

# Resume from here
next_action = find_next_pending_action(roadmap, current_week)
```

---

## 🔄 Rerouting Logic

### When Does Rerouting Trigger?

**Scenario 1: Confidence Drops Below 40**
```
Progress: Week 5, Confidence: 40 → 39 (after failed action)
         ↓
Condition: confidence < 40
         ↓
ReroutingAgent suggests SAFER roles:
  - More foundational level
  - Different career path (same domain)
  - Examples: Junior Engineer, Associate roles
         ↓
User Options:
  [1] Switch to suggested role → Regenerate roadmap
  [2] Continue current path → Keep trying
  [3] Return to previous role (if available)
```

**Scenario 2: Confidence Exceeds 80**
```
Progress: Week 15, Confidence: 75 → 81 (after successful action)
         ↓
Condition: confidence ≥ 80
         ↓
ReroutingAgent suggests ADVANCED roles:
  - More challenging career path
  - Leadership/specialized positions
  - Examples: Tech Lead, Solutions Architect
         ↓
User Options:
  [1] Switch to advanced role → Regenerate roadmap
  [2] Continue current path
  [3] Return to ORIGINAL target role (if different)
```

### Rerouting State Management

```json
{
  "reroute_state": {
    "is_active": false,
    "reroute_count": 0,
    "original_roadmap_id": "roadmap_123",
    "reroute_reason": null,
    "reroute_options": [],
    "selected_option": null,
    "rerouted_at": null,
    "can_return_to_previous": true
  }
}
```

### Rerouting Changes

When user accepts reroute suggestions:
1. **Save** original roadmap ID
2. **Mark** `reroute_state.is_active = true`
3. **Generate** new roadmap with suggested role
4. **Reset** action counter but **preserve** confidence score
5. **Maintain** ability to return to previous path

---

## 📁 File Structure

```
Nexus-AI/
├── 📄 README.md                          # Quick start guide
├── 📄 COMPREHENSIVE_README.md            # This file
├── 📄 FLOW_RESTRUCTURING_COMPLETE.md     # Flow architecture
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .env                               # Configuration (secrets)
├── 📄 LICENSE                            # MIT License
│
├── 📂 agent/
│   └── 🐍 agentic_career_navigator.py    # Main orchestrator (1692 lines)
│       ├── call_llm()                    # Shared LLM endpoint
│       ├── extract_json()                # JSON parsing utility
│       ├── ResumeAnalyzerAgent           # Resume processing
│       ├── ReadinessAssessmentAgent      # Skill evaluation
│       ├── MarketIntelligenceAgent       # Market analysis
│       ├── RoadmapAgent                  # Roadmap generation
│       ├── ActionAssessmentAgent         # Action evaluation
│       ├── ReroutingAgent                # Alternative paths
│       ├── FeedbackAgent                 # Progress reports
│       └── Orchestrator                  # Main coordinator
│
├── 📂 database/
│   ├── 🐍 __init__.py                    # Package exports
│   ├── 🐍 user_context.py                # Context manager (JSON storage)
│   └── 🐍 mongo_store.py                 # MongoDB sync layer
│
├── 📂 data/
│   ├── 📂 resumes/                       # Uploaded resume archives
│   │   └── user_{id}_{filename}.pdf
│   │
│   └── 📂 user_contexts/                 # JSON context backups
│       └── user_{id}_context.json
│
└── 📂 .git/                              # Version control

```

### Important Files

| File | Lines | Purpose |
|------|-------|---------|
| `agent/agentic_career_navigator.py` | 1692 | Main system with all 7 agents |
| `database/user_context.py` | 279 | Context manager for JSON storage |
| `database/mongo_store.py` | 239 | MongoDB sync & cloud storage |
| `.env` | 20 | Configuration & API keys |
| `requirements.txt` | 15 | Python dependencies |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- MongoDB Atlas account (optional but recommended)
- Groq API key (free at https://console.groq.com)
- Tesseract OCR (for image resume processing)

### Step 1: Clone & Install

```bash
git clone <repo-url>
cd Nexus-AI
pip install -r requirements.txt
```

### Step 2: Configure Environment

Create `.env` file:
```bash
# Groq API Configuration
GROQ_API_KEY=your-groq-api-key-here

# MongoDB Configuration (Optional)
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGO_DB=career_navigation
MONGO_COLL=user_contexts

# Resume Processing
RESUME_UPLOAD_DIR=./data/resumes
USER_CONTEXT_DIR=./data/user_contexts

# System Config
DEBUG=false
LOG_LEVEL=INFO
```

### Step 3: Create Directories

```bash
mkdir -p data/resumes data/user_contexts
```

### Step 4: Run System

```bash
python agent/agentic_career_navigator.py
```

### Step 5: View Data (Optional)

**Using MongoDB Compass:**
1. Open MongoDB Compass
2. Connect with your MONGO_URI
3. Navigate to: `career_navigation` → `user_contexts`
4. View all stored user profiles

---

## 📊 Data Flow Visualization

### New User → Complete Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEW USER STARTS                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌─ Resume Uploaded (Archive)
         │
    SKILL INPUT ─┴─ Manual Entry (Typed)
         │
         ▼ Extract/Normalize Skills
    
    ┌──────────────────────────────┐
    │ context["profile"]["skills"] │
    └──────────────────────────────┘
         │
         ▼ User selects target role
    
    ┌───────────────────────────────────────────┐
    │ context["career_state"]                   │
    │ current_target_role = "Senior BE Dev"    │
    └───────────────────────────────────────────┘
         │
         ▼ ReadinessAssessmentAgent (10 Q&A)
    
    ┌──────────────────────────────────────────────┐
    │ context["readiness_assessment"]              │
    │ - confidence_score = 75                      │
    │ - status = "qualified"                       │
    │ - matched_skills = [...]                     │
    │ - alternative_roles = [...]                  │
    └──────────────────────────────────────────────┘
         │
         ├─→ MarketIntelligenceAgent
         │   └─ market_analysis saved
         │
         ▼ RoadmapAgent (5 months × 4 actions)
    
    ┌──────────────────────────────────────────┐
    │ context["active_roadmap"]["phases"]       │
    │ [                                        │
    │   {month: 1, actions: [4 weeks]},        │
    │   {month: 2, actions: [4 weeks]},        │
    │   ...                                     │
    │   {month: 5, actions: [4 weeks]}         │
    │ ]                                         │
    │ Total: 20 actions                        │
    └──────────────────────────────────────────┘
         │
         ├─────────────────────────────────────┐
         │  ACTION LOOP (Weeks 1-20)           │
         ▼                                       │
    ActionAssessmentAgent × 20                  │
         │ Each action:                        │
         │ ├─ Score 0-100                      │
         │ ├─ Confidence +/-1                  │
         │ └─ Auto-rerouting check             │
         │                                      │
         ▼                                       │
    ┌────────────────────────────────────┐      │
    │ context["progress"]                │      │
    │ - weeks_completed = 4              │      │
    │ - actions_completed = 4            │      │
    │ - current_streak = 1 week          │      │
    │ - confidence_score = 76            │      │
    └────────────────────────────────────┘      │
         └─────────────────────────────────────┘
         │
         ▼ FeedbackAgent (Final Report)
    
    ┌──────────────────────────────────────────┐
    │ FINAL FEEDBACK                           │
    │ - Progress summary                       │
    │ - Strengths & gaps                       │
    │ - Recommendations                        │
    │ - Motivational message                   │
    └──────────────────────────────────────────┘
         │
         ▼ PERSIST TO DATABASE
    
    ┌──────────────────────────────────────────┐
    │ JSON: ./data/user_contexts/{id}.json     │
    │ MongoDB: career_navigation.user_contexts │
    └──────────────────────────────────────────┘
         │
         ▼
    ✅ ALL DATA SAVED
```

---

## 🎯 Key Metrics Tracked

### Confidence Score
- **Range**: 0-100
- **Starts**: Equal to readiness_score
- **Updates**: +1 per passed action, -1 per failed action
- **Triggers Rerouting**: < 40 or ≥ 80

### Progress Metrics
```json
{
  "weeks_completed": 4,           // Out of 20
  "actions_completed": 4,         // Passed
  "actions_failed": 0,            // Failed
  "current_streak_weeks": 4,      // Consecutive weeks
  "total_hours_invested": 32,     // Estimated
  "completion_percentage": 20     // 4/20 weeks
}
```

### Skill Match
```json
{
  "skill_match_percentage": 85,   // How many target skills user has
  "matched_skills": ["Python", "Docker"],
  "missing_skills": ["Kubernetes", "gRPC"],
  "surplus_skills": ["Fortran"]   // Has but doesn't need
}
```

---

## 🔐 Security & Privacy

- **API Keys**: Stored in `.env` (git-ignored)
- **Resume Files**: Archived locally in `./data/resumes/`
- **User Data**: Encrypted in MongoDB Atlas
- **No Cookies/Tracking**: Stateless session-based
- **Data Ownership**: User owns all their data

---

## 📚 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| LLM | Groq API | openai/gpt-oss-120b |
| Database | MongoDB Atlas | Cloud |
| Local Storage | JSON | File-based |
| PDF Processing | pdfminer.six | ≥20220524 |
| OCR | pytesseract + Tesseract | ≥0.3.10 |
| Image Processing | Pillow | ≥10.0.0 |
| Python | 3.9+ | Latest |

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test locally
5. Submit pull request

---

## 📝 License

MIT License - see LICENSE file

---

## 🎓 Learning Resources

- **Groq Documentation**: https://console.groq.com/docs
- **MongoDB Manual**: https://docs.mongodb.com/manual/
- **Python Async**: https://docs.python.org/3/library/asyncio.html

---

## ✅ Checklist for First Run

- [ ] Python 3.9+ installed
- [ ] `pip install -r requirements.txt` completed
- [ ] `.env` file configured with GROQ_API_KEY
- [ ] `./data/resumes` and `./data/user_contexts` directories created
- [ ] MongoDB URI set (optional)
- [ ] `python agent/agentic_career_navigator.py` runs without errors
- [ ] Sample resume located at `./sample_resume.pdf`
- [ ] User creates account successfully
- [ ] Completes readiness assessment
- [ ] Roadmap generates (5 months × 4 actions)

---

**Last Updated**: February 17, 2026
**System Status**: ✅ Production Ready
**MongoDB Connection**: ✅ Connected & Syncing
**Groq API**: ✅ Integrated
