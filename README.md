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
| Auth | bcrypt password hashing, SMTP email delivery |
| Resume parsing | PyMuPDF, Pillow, pytesseract |

---

## Project Structure

```
Nexus-AI/
├── .env.example                         # Root env template
├── docker-compose.yml                   # Production containers
├── docker-compose.dev.yml               # Development containers (live reload)
├── backend/
│   ├── Dockerfile
│   ├── main.py                          # FastAPI app — all endpoints
│   ├── requirements.txt
│   ├── agents/
│   │   ├── agentic_career_navigator.py  # All 6 agent classes
│   │   ├── orchestrator_wrapper.py      # HTTP ↔ agent translation layer
│   │   └── hands_on_agent.py
│   ├── core/
│   │   ├── auth.py                      # Password hashing + email sending
│   │   └── user_context.py
│   └── database/
│       ├── db.py                        # MongoDB Atlas singleton
│       └── schemas.py                   # Pydantic v2 models
├── frontend/
│   ├── Dockerfile                       # Production image
│   ├── Dockerfile.dev                   # Dev image (next dev + hot reload)
│   └── src/
│       ├── app/
│       │   ├── page.tsx                 # Landing page
│       │   ├── auth/                    # Email + password auth
│       │   ├── onboarding/              # Profile setup + resume upload
│       │   ├── readiness/               # 10-question AI assessment
│       │   ├── result/                  # Score gauge + adjacent roles
│       │   ├── dashboard/               # Central hub with charts
│       │   ├── roadmap/                 # 5-month accordion + assessment modal
│       │   ├── market/                  # Demand scores, salary, companies
│       │   ├── hands-on/                # AI mentor chat
│       │   ├── reroute/                 # Confidence-based path adjustment
│       │   ├── feedback/                # Full progress analytics
│       │   ├── help/                    # Help center
│       │   └── credits/                 # Team & contact
│       ├── components/
│       │   ├── ParticleBackground.tsx
│       │   ├── HeroParticleBackground.tsx
│       │   └── AIThinkingOverlay.tsx
│       ├── lib/api.ts                   # All backend API calls (Axios)
│       └── store/useStore.ts            # Zustand global state
└── data/
    ├── resumes/
    └── user_contexts/
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- [MongoDB Atlas](https://cloud.mongodb.com) account (free M0 tier)
- [Groq API key](https://console.groq.com)
- Gmail account with an [App Password](https://myaccount.google.com/apppasswords) for email delivery

---

## Setup

### 1 — MongoDB Atlas

1. Sign up at https://cloud.mongodb.com
2. Create a free **M0** cluster
3. **Database Access** → Add user → Password auth → *Read and write to any database*
4. **Network Access** → Allow Access from Anywhere (`0.0.0.0/0`)
5. **Database** → Connect → Drivers → Python 4.7+ → copy the connection string

### 2 — Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# MongoDB Atlas
MONGO_URI=mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/
MONGO_DB=career_navigation
MONGO_COLL=user_contexts

# CORS
FRONTEND_URL=https://yourdomain.com

# Backend URL as seen by the browser
NEXT_PUBLIC_API_URL=https://api.yourdomain.com

# Email (Gmail App Password)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=youraddress@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
SMTP_FROM=youraddress@gmail.com
```

### 3 — Run

**Production:**
```bash
docker compose up --build
```

**Development (live reload — no rebuild on code changes):**
```bash
docker compose -f docker-compose.dev.yml up --build
```

---

## After Code Changes

| Mode | What to do |
|---|---|
| Production | `docker compose up --build` |
| Development | Just save the file — reloads automatically |

```bash
# Stop production
docker compose down

# Stop development
docker compose -f docker-compose.dev.yml down

# Wipe volumes (clears uploaded data)
docker compose down -v
```

---

## Authentication Flow

1. User enters email → backend checks if account exists
   - **New email** → account created, random password generated + emailed
   - **Existing email** → prompts for password
2. User enters password → verified against bcrypt hash in MongoDB
3. Correct password → redirected to onboarding (new) or dashboard (existing)
4. **Forgot password** → new password generated and emailed, hash updated

---

## User Flow

```
Landing (/)
  └─► Sign In (/auth)
        ├─► Onboarding    (new user)
        │     Step 1: Profile
        │     Step 2: Skills — manual entry OR resume upload (PDF/PNG/JPG)
        │             Auto-extracts skills, strengths, weaknesses
        │     Step 3: Target Role
        │         └─► Readiness Assessment → 10 AI questions
        │                 └─► Result → Score + adjacent roles
        │                       └─► Dashboard
        └─► Dashboard     (existing user)
              ├─► Roadmap      5 months × 4 actions, assessment modal
              ├─► Market       Demand score, salary, companies
              ├─► Hands-On     AI mentor chat
              ├─► Reroute      Confidence-based path adjustment
              ├─► Feedback     Full analytics
              ├─► Help         Help center
              └─► Credits      Team & contact
```

---

## AI Agents

| Agent | What it does |
|---|---|
| `ReadinessAssessmentAgent` | Generates 10 questions, scores answers fairly |
| `MarketIntelligenceAgent` | Demand score, salary range, competition, adjacent roles |
| `RoadmapAgent` | Builds 5-month × 4-action personalised roadmap |
| `ActionAssessmentAgent` | Generates 10 questions per action, evaluates answers, updates confidence |
| `ReroutingAgent` | Checks confidence threshold, suggests role adjustments |
| `FeedbackAgent` | Full progress report with insights and adjustments |

**Confidence:** starts at readiness score · `+1` per passed action · `-1` per failed · clamped `[0, 100]`  
**Reroute trigger:** confidence `< 40` → suggest safer roles · `≥ 80` → allow return to previous role

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/login` | Check email, send password if new user |
| POST | `/api/auth/verify` | Verify password, return user_id |
| POST | `/api/auth/forgot-password` | Generate + email new password |
| POST | `/api/onboard` | Create/update user profile + market intel |
| POST | `/api/readiness/start` | Generate 10 readiness questions |
| POST | `/api/readiness/evaluate` | Score answers + generate roadmap |
| GET | `/api/dashboard/{uid}` | Aggregate all user state |
| GET | `/api/roadmap/{uid}` | Return active roadmap |
| POST | `/api/roadmap/regenerate` | Regenerate roadmap |
| POST | `/api/action/questions` | Generate 10 questions for an action |
| POST | `/api/action/assess` | Score answers + update confidence |
| GET | `/api/market/{uid}` | Return market analysis |
| POST | `/api/reroute` | Analyse rerouting + optional role switch |
| POST | `/api/feedback` | Generate feedback report |
| POST | `/api/hands-on/chat` | Stateless AI mentor chat |
| POST | `/api/resume/upload` | Upload PDF/image resume |
| GET | `/health` | Health check |

---

## Architecture Notes

- All LLM calls are **server-side only** — the frontend never calls Groq directly
- Passwords are hashed with **bcrypt** — plain text is never stored
- Q&A pairs are **never persisted** — only scores and summaries go to MongoDB
- `orchestrator_wrapper.py` is a pure HTTP ↔ agent translation layer — no business logic
- `next.config.js` uses `output: "standalone"` for minimal production Docker images

---

## Credits

Built by the Nexus-AI team — [deepakthillaikannu@gmail.com](mailto:deepakthillaikannu@gmail.com)
