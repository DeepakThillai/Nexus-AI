# Nexus-AI: Agentic Career Navigator
### Full-Stack · FastAPI + Next.js 14 · MongoDB Atlas · Groq LLM

Nexus-AI deploys specialised AI agents to assess your career readiness, generate a personalised 5-month roadmap, track your confidence, surface market intelligence, and reroute you when momentum drops — all in real time.

---

## Screenshots

### Landing Page
![Landing page](images/first_page.png)

### Dashboard
![Dashboard](images/dashboard.png)

### Readiness Assessment

| Assessment | Readiness Score |
|:---:|:---:|
| ![Assessment](images/assessment.png) | ![Score](images/readiness_score.png) |

### Personalised Roadmap
![Roadmap](images/roadmap.png)

### Market Intelligence
![Market report](images/market_report.png)

### Hands-On AI Mentor

| Practical Task | Guided Mentoring |
|:---:|:---:|
| ![Task](images/handson1.png) | ![Mentoring](images/handson2.png) |

### Career Rerouting & Feedback

| Rerouting | Feedback |
|:---:|:---:|
| ![Reroute](images/reroute.png) | ![Feedback](images/feedback.png) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS, Framer Motion, Zustand, Recharts |
| Backend | FastAPI, Python 3.10+, Uvicorn |
| AI | Groq API (RAG)) |
| Database | MongoDB Atlas (cloud, free tier works) |
| Resume parsing | pdfminer.six, PyPDF2, Pillow, pytesseract |

---

## Project Structure

```
Nexus-AI/
├── backend/
│   ├── main.py                        # FastAPI app — all endpoints
│   ├── .env                           # Your secrets (not committed)
│   ├── .env.example                   # Template
│   ├── requirements.txt
│   ├── agents/
│   │   ├── agentic_career_navigator.py  # All 6 agent classes
│   │   ├── orchestrator_wrapper.py      # HTTP ↔ agent translation layer
│   │   ├── hands_on_agent.py            # CLI hands-on mentor (standalone)
│   │   └── __init__.py
│   ├── core/
│   │   └── user_context.py              # JSON-based context helpers
│   └── database/
│       ├── db.py                        # MongoDB Atlas singleton
│       ├── schemas.py                   # Pydantic v2 request/response models
│       └── mongo_store.py               # Legacy store (not used by API)
├── frontend/
│   └── src/
│       ├── app/                         # Next.js App Router pages
│       │   ├── page.tsx                 # Landing page
│       │   ├── auth/                    # Email-based auth
│       │   ├── onboarding/              # 3-step profile setup + resume upload
│       │   ├── readiness/               # 10-question AI assessment
│       │   ├── result/                  # Score gauge + adjacent roles
│       │   ├── dashboard/               # Central hub with charts
│       │   ├── roadmap/                 # 5-month accordion + assessment modal
│       │   ├── market/                  # Demand scores, salary, companies
│       │   ├── hands-on/                # AI mentor chat
│       │   ├── reroute/                 # Confidence-based path adjustment
│       │   └── feedback/                # Full progress analytics
│       ├── components/
│       │   ├── ParticleBackground.tsx   # Cyan particle canvas (inner pages)
│       │   ├── HeroParticleBackground.tsx # White particles (landing)
│       │   └── AIThinkingOverlay.tsx    # Full-screen AI loading overlay
│       ├── lib/
│       │   └── api.ts                   # All backend API calls (Axios)
│       └── store/
│           └── useStore.ts              # Zustand global state (persisted)
└── data/
    ├── resumes/                         # Uploaded resume files
    └── user_contexts/                   # JSON context snapshots
```

---

## Quick Start

You need **3 terminals** running simultaneously.

```
Terminal 1 → MongoDB Atlas  (cloud — no local install)
Terminal 2 → Backend        (FastAPI on :8000)
Terminal 3 → Frontend       (Next.js on :3000)
```

---

## Docker (recommended for deployment)

The entire stack runs with a single command.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/install/) installed
- A MongoDB Atlas connection string (free tier — see Step 1 below)
- A Groq API key from https://console.groq.com

### 1 — Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` at the project root:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/
MONGO_DB=career_navigation
MONGO_COLL=user_contexts
FRONTEND_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2 — Build and run

```bash
docker compose up --build
```

First build takes ~3–5 minutes (installs all deps, builds Next.js). Subsequent starts are instant.

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Health check | http://localhost:8000/health |

### Useful commands

```bash
# Run in background
docker compose up --build -d

# View logs
docker compose logs -f

# View logs for one service
docker compose logs -f backend
docker compose logs -f frontend

# Stop everything
docker compose down

# Stop and remove volumes (clears uploaded resumes)
docker compose down -v

# Rebuild a single service after code changes
docker compose up --build backend
docker compose up --build frontend
```

---

## Step 1 — MongoDB Atlas (free, ~5 min)

1. Go to https://cloud.mongodb.com → sign up / log in
2. **Build a Database** → choose **M0 Free** tier → Create
3. **Database Access** → Add New Database User → Password auth → Role: *Read and write to any database*
4. **Network Access** → Add IP Address → **Allow Access from Anywhere** (`0.0.0.0/0`)
5. **Database** → Connect → Drivers → Python 3.6+ → copy the connection string:
   ```
   mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/
   ```

---

## Step 2 — Backend Setup

```bash
# From the project root (d:\Nexus-AI or ~/Nexus-AI)
cd backend

# Copy and fill in the env file
cp .env.example .env
```

Edit `backend/.env`:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_URI=mongodb+srv://youruser:yourpassword@cluster0.xxxxx.mongodb.net/
MONGO_DB=career_navigation
MONGO_COLL=user_contexts
FRONTEND_URL=http://localhost:3000
RESUME_UPLOAD_DIR=./data/resumes
USER_CONTEXT_DIR=./data/user_contexts
```

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend — run from the PROJECT ROOT, not from inside backend/
cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

> ⚠️ **Important:** always run `uvicorn` from `d:\Nexus-AI` (the project root), not from inside `backend/`. The app uses `backend.main:app` package imports.

You should see:
```
[DB] Connected to MongoDB Atlas -> career_navigation.user_contexts
╔══════════════════════════════════════════╗
║   Nexus-AI Backend Starting...           ║
╚══════════════════════════════════════════╝
  MongoDB: ✓ Connected
  GROQ_API_KEY: ✓ Set
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Health check: http://localhost:8000/health → `{"status":"ok","mongo":true,"groq_key_set":true}`

---

## Step 3 — Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open http://localhost:3000

> The frontend reads `NEXT_PUBLIC_API_URL` from environment (defaults to `http://localhost:8000`). No `.env.local` changes needed for local development.

---

## User Flow

```
/ (Landing)
  └─► /auth
        ├─► /onboarding    (new user)  → 3 steps: Profile → Skills → Target Role
        │         └─► /readiness       → 10 AI questions, one per screen
        │                   └─► /result → Score gauge + adjacent roles
        │                         └─► /dashboard
        └─► /dashboard     (existing user)
              ├─► /roadmap     5-month accordion, 4 actions/month
              │                 "Start Assessment" → 10-question modal
              │                 Score saved → confidence updated
              ├─► /market      Demand score, salary range, hiring companies
              ├─► /hands-on    AI mentor chat (practical task-based)
              ├─► /reroute     Confidence-based path adjustment + role switch
              └─► /feedback    Full analytics from FeedbackAgent
```

---

## AI Agents

| Agent | Trigger | What it does |
|---|---|---|
| `ReadinessAssessmentAgent` | `/api/readiness/start` + `/evaluate` | Generates 10 role-specific questions, scores all answers |
| `MarketIntelligenceAgent` | On onboarding + role switch | Demand score, salary, competition, adjacent roles |
| `RoadmapAgent` | After readiness evaluation | Builds 5-month × 4-action personalised roadmap |
| `ActionAssessmentAgent` | Per roadmap action | Generates 10 questions, evaluates answers, updates confidence |
| `ReroutingAgent` | After every action + `/api/reroute` | Checks confidence threshold, suggests safer/advanced roles |
| `FeedbackAgent` | `/api/feedback` | Generates full progress report with insights and adjustments |

**Confidence scoring:** starts at readiness score, `+1` per passed action, `-1` per failed, clamped `[0, 100]`.  
**Reroute trigger:** confidence `< 40` → suggest safer roles. Confidence `≥ 80` → allow return to previous role.

---

## API Reference

| Method | Endpoint | Agent called |
|---|---|---|
| POST | `/api/onboard` | `MarketIntelligenceAgent` |
| POST | `/api/readiness/start` | `ReadinessAssessmentAgent._generate_questions` |
| POST | `/api/readiness/evaluate` | `ReadinessAssessmentAgent._evaluate` + `RoadmapAgent` |
| GET | `/api/dashboard/{uid}` | Aggregates MongoDB state |
| GET | `/api/roadmap/{uid}` | Returns `active_roadmap` from MongoDB |
| POST | `/api/roadmap/regenerate` | `RoadmapAgent.run` |
| POST | `/api/action/questions` | `ActionAssessmentAgent._generate_questions` |
| POST | `/api/action/assess` | `ActionAssessmentAgent._evaluate` + confidence update |
| GET | `/api/market/{uid}` | Returns cached `market_analysis` from MongoDB |
| POST | `/api/reroute` | `ReroutingAgent.run` + optional role switch |
| POST | `/api/feedback` | `FeedbackAgent.run` |
| POST | `/api/hands-on/chat` | Groq LLM stateless chat |
| POST | `/api/resume/upload` | `ResumeAnalyzerAgent` (PDF/PNG/JPG) |
| POST | `/api/resume/extract-skills` | `ResumeAnalyzerAgent` (raw text) |

---

## Architecture Notes

- All LLM calls are **server-side only** — the frontend never touches the Groq API directly
- Q&A pairs are **never persisted** — only scores and summaries go to MongoDB
- Questions are cached in-memory per session (`_session_cache`); if the server restarts mid-session they are auto-regenerated from the action title
- The 6 agent classes in `agentic_career_navigator.py` are self-contained and not modified by the API layer
- `orchestrator_wrapper.py` is a pure translation layer between HTTP and agents — no business logic

---

## Troubleshooting

**`GET / 404` on frontend**  
→ Make sure there is no empty `frontend/app/` directory. If it exists, delete it — Next.js will prefer it over `src/app/` and serve 404 for everything.

**`ModuleNotFoundError: No module named 'backend'`**  
→ You are running `uvicorn` from inside the `backend/` folder. Run it from the project root instead:
```bash
# From d:\Nexus-AI (not d:\Nexus-AI\backend)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**`[DB] Config error: MONGO_URI is not set`**  
→ Your `.env` file is in `backend/.env` but you're running Python from a different directory. This is already fixed — `db.py` resolves the `.env` path relative to its own file location.

**`[DB] Atlas connection failed`**  
→ Check your `MONGO_URI` — password must be URL-encoded if it contains special characters (`@`, `#`, `%`, etc.)  
→ Check Atlas **Network Access** — your IP must be whitelisted or use `0.0.0.0/0` for development

**`GROQ_API_KEY: ✗ MISSING`**  
→ Make sure `backend/.env` exists and contains `GROQ_API_KEY=gsk_...`

**`pymongo.errors.ConfigurationError: ... dnspython`**  
→ Run: `pip install "pymongo[srv]"` — the `[srv]` extra installs `dnspython` required for Atlas connection strings

**Resume upload fails**  
→ `pytesseract` requires the Tesseract OCR binary installed separately — it is not a Python package. Download from https://github.com/UB-Mannheim/tesseract/wiki (Windows) or `sudo apt install tesseract-ocr` (Linux). PDF uploads work without it.

**Assessment questions not loading**  
→ If the server restarted between `/api/action/questions` and `/api/action/assess`, the session cache is gone. The backend auto-regenerates questions from the action title — this is handled gracefully.
