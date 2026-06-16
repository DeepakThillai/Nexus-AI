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
| AI | Groq API (LLaMA-class models) |
| Database | MongoDB Atlas (cloud, free tier works) |
| Resume parsing | PyMuPDF, Pillow, pytesseract |

---

## Project Structure

```
Nexus-AI/
├── .env.example                         # Root env template for Docker
├── docker-compose.yml                   # Production containers
├── docker-compose.dev.yml               # Development containers (live reload)
├── backend/
│   ├── Dockerfile
│   ├── main.py                          # FastAPI app — all endpoints
│   ├── requirements.txt
│   ├── agents/
│   │   ├── agentic_career_navigator.py  # All 6 agent classes
│   │   ├── orchestrator_wrapper.py      # HTTP ↔ agent translation layer
│   │   └── hands_on_agent.py            # Standalone CLI mentor
│   ├── core/
│   │   └── user_context.py
│   └── database/
│       ├── db.py                        # MongoDB Atlas singleton
│       └── schemas.py                   # Pydantic v2 models
├── frontend/
│   ├── Dockerfile                       # Production image
│   ├── Dockerfile.dev                   # Dev image (next dev)
│   └── src/
│       ├── app/
│       │   ├── page.tsx                 # Landing page
│       │   ├── auth/                    # Email-based auth
│       │   ├── onboarding/              # Profile setup + resume upload
│       │   ├── readiness/               # 10-question AI assessment
│       │   ├── result/                  # Score gauge + adjacent roles
│       │   ├── dashboard/               # Central hub with charts
│       │   ├── roadmap/                 # 5-month accordion + assessment modal
│       │   ├── market/                  # Demand scores, salary, companies
│       │   ├── hands-on/                # AI mentor chat
│       │   ├── reroute/                 # Confidence-based path adjustment
│       │   ├── feedback/                # Full progress analytics
│       │   ├── help/                    # Help center (7 topics)
│       │   └── credits/                 # Team & contact
│       ├── components/
│       │   ├── ParticleBackground.tsx
│       │   ├── HeroParticleBackground.tsx
│       │   └── AIThinkingOverlay.tsx
│       ├── lib/api.ts                   # All backend API calls (Axios)
│       └── store/useStore.ts            # Zustand global state
└── data/
    ├── resumes/                         # Uploaded resume files
    └── user_contexts/                   # JSON context snapshots
```

---

## Docker Setup (recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- MongoDB Atlas connection string (free tier — see MongoDB section below)
- Groq API key from https://console.groq.com

### 1 — Create your `.env`

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/
MONGO_DB=career_navigation
MONGO_COLL=user_contexts
FRONTEND_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2 — Run

**Production** — code baked into image, use for deployment:
```bash
docker compose up --build
```

**Development** — live reload, no rebuild needed on code changes:
```bash
docker compose -f docker-compose.dev.yml up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Health check | http://localhost:8000/health |

### After code changes

| Mode | What to do |
|---|---|
| Production | `docker compose up --build` |
| Development | Just save the file — reloads automatically |

### Other commands

```bash
# Run production in background
docker compose up --build -d

# Stream logs
docker compose logs -f

# Stop
docker compose down

# Stop and wipe data volumes
docker compose down -v

# Stop dev containers
docker compose -f docker-compose.dev.yml down
```

---

## Manual Setup (without Docker)

### MongoDB Atlas

1. Go to https://cloud.mongodb.com → sign up / log in
2. **Build a Database** → **M0 Free** → Create
3. **Database Access** → Add user → Password auth → Role: *Read and write to any database*
4. **Network Access** → Add IP → **Allow Access from Anywhere** (`0.0.0.0/0`)
5. **Database** → Connect → Drivers → Python 3.6+ → copy connection string

### Backend

```bash
# From the project root
pip install -r backend/requirements.txt

# Always run from project root — not from inside backend/
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Create `backend/.env`:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/
MONGO_DB=career_navigation
MONGO_COLL=user_contexts
FRONTEND_URL=http://localhost:3000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## User Flow

```
/ (Landing)
  └─► /auth
        ├─► /onboarding    (new user)
        │     Step 1: Profile (name, email, phone, experience)
        │     Step 2: Skills — manual entry OR upload resume (PDF/PNG/JPG)
        │             Resume auto-extracts skills, strengths, weaknesses
        │     Step 3: Target Role
        │         └─► /readiness  → 10 AI questions, one per screen
        │                 └─► /result → Score gauge + adjacent roles
        │                       └─► /dashboard
        └─► /dashboard     (existing user)
              ├─► /roadmap     5-month accordion, 4 actions/month
              │                 "Start Assessment" → 10-question modal
              │                 Score saved → confidence ±1
              ├─► /market      Demand score, salary, hiring companies
              ├─► /hands-on    AI mentor chat (task-based learning)
              ├─► /reroute     Confidence-based path adjustment
              ├─► /feedback    Full analytics report
              ├─► /help        Help center
              └─► /credits     Team info & contact
```

---

## AI Agents

| Agent | Triggered by | What it does |
|---|---|---|
| `ReadinessAssessmentAgent` | `/api/readiness/start` + `/evaluate` | Generates 10 questions, scores answers fairly |
| `MarketIntelligenceAgent` | Onboarding + role switch | Demand score, salary range, competition, adjacent roles |
| `RoadmapAgent` | After readiness evaluation | Builds 5-month × 4-action personalised roadmap |
| `ActionAssessmentAgent` | Per roadmap action | Generates 10 questions, evaluates answers, updates confidence |
| `ReroutingAgent` | After every action + `/api/reroute` | Checks confidence threshold, suggests role adjustments |
| `FeedbackAgent` | `/api/feedback` | Generates full progress report with insights |

**Confidence scoring:** initialises at readiness score · `+1` per passed action · `-1` per failed · clamped `[0, 100]`  
**Reroute trigger:** confidence `< 40` → suggest safer roles · confidence `≥ 80` → allow return to previous role  
**Evaluation:** assessors give benefit of the doubt — partial answers and genuine attempts are scored positively

---

## Resume Upload

On the Skills step of onboarding, users can upload a PDF or image resume instead of entering skills manually.

- **Extraction:** PyMuPDF extracts text from PDF; pytesseract handles image-based resumes
- **AI parsing:** Single Groq call returns `skills`, `strengths`, `weaknesses`, `soft_skills`, `name`, `phone`, `experience_years`
- **Auto-fill:** All three fields (skills, strengths, weaknesses) are populated automatically — same as manual entry
- **Editable:** User can remove or add tags before proceeding

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/onboard` | Create/update user + generate market intel |
| POST | `/api/readiness/start` | Generate 10 readiness questions |
| POST | `/api/readiness/evaluate` | Score answers + generate roadmap |
| GET | `/api/dashboard/{uid}` | Aggregate all user state |
| GET | `/api/roadmap/{uid}` | Return active roadmap |
| POST | `/api/roadmap/regenerate` | Regenerate roadmap for role |
| POST | `/api/action/questions` | Generate 10 questions for an action |
| POST | `/api/action/assess` | Score answers + update confidence |
| GET | `/api/market/{uid}` | Return market analysis |
| POST | `/api/reroute` | Analyse rerouting + optional role switch |
| POST | `/api/feedback` | Generate feedback report |
| POST | `/api/hands-on/chat` | Stateless AI mentor chat |
| POST | `/api/resume/upload` | Upload PDF/image resume → extract skills |
| POST | `/api/resume/extract-skills` | Extract skills from raw text |
| GET | `/health` | Health check |

---

## Architecture Notes

- All LLM calls are **server-side only** — the frontend never calls Groq directly
- Q&A pairs are **never persisted** — only scores and summaries go to MongoDB
- Session question cache is in-memory; auto-regenerated from action title if the server restarts mid-session
- `orchestrator_wrapper.py` is a pure HTTP ↔ agent translation layer — no business logic
- `next.config.js` uses `output: "standalone"` for minimal production Docker images

---

## Credits

Built by the Nexus-AI team. Visit `/credits` in the app or contact **deepakthillaikannu@gmail.com**.
