# Nexus-AI: Agentic Career Navigator
## Full-Stack · FastAPI + Next.js 14 · MongoDB Atlas

---

## Application Screenshots

### Landing Page

![Nexus-AI landing page](images/first_page.png)

### Dashboard

![Nexus-AI dashboard](images/dashboard.png)

### Readiness Assessment

| Assessment | Readiness Score |
|:---:|:---:|
| ![Readiness assessment](images/assessment.png) | ![Readiness score](images/readiness_score.png) |

### Personalized Roadmap

![Personalized career roadmap](images/roadmap.png)

### Market Intelligence

![Career market report](images/market_report.png)

### Hands-On AI Mentor

| Practical Task | Guided Mentoring |
|:---:|:---:|
| ![Hands-on practical task](images/handson1.png) | ![Hands-on guided mentoring](images/handson2.png) |

### Career Rerouting and Feedback

| Career Rerouting | Performance Feedback |
|:---:|:---:|
| ![Career rerouting](images/reroute.png) | ![Performance feedback](images/feedback.png) |

---

## Quick Start (3 terminals)

```
Terminal 1 → MongoDB Atlas  (cloud — no local install needed)
Terminal 2 → Backend  (FastAPI on :8000)
Terminal 3 → Frontend (Next.js on :3000)
```

---

## Step 1 — MongoDB Atlas Setup (free, ~5 min)

You do NOT need to install MongoDB locally. Atlas is fully cloud-hosted.

### 1.1 Create free cluster
1. Go to https://cloud.mongodb.com and sign up / log in
2. Click **"Build a Database"** → choose **M0 Free** tier
3. Select any cloud provider + region → click **Create**

### 1.2 Create a database user
1. In the left sidebar → **Database Access** → **Add New Database User**
2. Choose **Password** authentication
3. Set username (e.g. `nexus_user`) and a strong password — **save both**
4. Role: **Read and write to any database** → click **Add User**

### 1.3 Whitelist your IP
1. Left sidebar → **Network Access** → **Add IP Address**
2. Click **"Allow Access from Anywhere"** (adds `0.0.0.0/0`) for development
3. Click **Confirm**

### 1.4 Get your connection string
1. Left sidebar → **Database** → click **Connect** on your cluster
2. Choose **"Drivers"** → Language: **Python** → Version: **3.6+**
3. Copy the connection string — it looks like:
   ```
   mongodb+srv://nexus_user:<password>@cluster0.xxxxx.mongodb.net/
   ```
4. Replace `<password>` with your actual password

---

## Step 2 — Backend Setup

```bash
cd backend

# Copy env file
cp .env.example .env
```

Edit `.env` — fill in your values:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_URI=mongodb+srv://nexus_user:YourPassword@cluster0.xxxxx.mongodb.net/
MONGO_DB=nexus_ai
MONGO_COLL=users
FRONTEND_URL=http://localhost:3000
```

```bash
# Install dependencies (pymongo[srv] handles the Atlas SRV format)
pip install -r requirements.txt

# Start backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
[DB] Connected to MongoDB Atlas -> nexus_ai.users
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify: http://localhost:8000/health → `{"status":"ok","mongo":true,"groq_key_set":true}`

---

## Step 3 — Frontend Setup

```bash
cd frontend

# Copy env file
cp .env.local.example .env.local
# File already contains: NEXT_PUBLIC_API_URL=http://localhost:8000

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open http://localhost:3000

---

## Full User Flow

```
/ (Landing)
  → /auth          Email entry → new user or existing user
  → /onboarding    3-step: Profile → Skills → Target Role
  → /readiness     10 GPT questions, 1 per screen
  → /result        Score gauge + adjacent roles
  → /dashboard     Stats cards + score chart (central hub)
      → /roadmap       5 months × 4 actions accordion
                       "Start Assessment" → 10-question modal
                       ActionAssessmentAgent evaluates → score saved
      → /market        Demand score, salary, companies, charts
      → /hands-on      AI mentor chat (task-based)
      → /reroute       Confidence circle + role switch
      → /feedback      Full analytics from FeedbackAgent
```

---

## Roadmap & Action Assessment Flow (fixed)

Each month has **exactly 4 actions** from the RoadmapAgent.

Clicking **"Start Assessment"** on any pending action:
1. Calls `POST /api/action/questions`
   → `ActionAssessmentAgent._generate_questions()` — GPT generates 10 questions
2. Shows questions one at a time (progress bar)
3. On final answer → `POST /api/action/assess`
   → `ActionAssessmentAgent._evaluate()` — GPT scores all answers
   → Score saved to MongoDB
   → `confidence_score ±1` updated in MongoDB
   → Rerouting check runs automatically
4. Result shown: score, pass/fail, updated confidence

**Score ≥ 50 = Passed (confidence +1)**
**Score < 50 = Failed (confidence -1)**

---

## API Reference

| Method | Endpoint | What calls |
|--------|----------|------------|
| POST | `/api/onboard` | Creates user + MarketIntelligenceAgent |
| POST | `/api/readiness/start` | ReadinessAssessmentAgent._generate_questions |
| POST | `/api/readiness/evaluate` | ReadinessAssessmentAgent._evaluate |
| GET  | `/api/dashboard/{uid}` | Aggregates MongoDB state |
| GET  | `/api/roadmap/{uid}` | Returns active_roadmap from MongoDB |
| POST | `/api/roadmap/regenerate` | RoadmapAgent.run |
| POST | `/api/action/questions` | ActionAssessmentAgent._generate_questions |
| POST | `/api/action/assess` | ActionAssessmentAgent._evaluate + confidence update |
| GET  | `/api/market/{uid}` | MarketIntelligenceAgent.run (cached) |
| POST | `/api/reroute` | ReroutingAgent.run + optional role switch |
| POST | `/api/feedback` | FeedbackAgent.run |
| POST | `/api/hands-on/chat` | HandsOnAgent pattern (stateless) |

---

## Architecture Rules (preserved)

- All 6 original agent classes are **UNMODIFIED**
- All GPT calls stay **server-side** (never in frontend)
- Q&A is **never persisted** — only scores and summaries go to MongoDB
- Confidence scoring is unchanged: `±1 per action, clamped to [0, 100]`
- Rerouting thresholds unchanged: `< 40 → suggest safer, ≥ 80 → allow return`
- MongoDB schema matches the original document structure exactly

---

## Troubleshooting

**`[DB] Atlas connection failed`**
→ Check MONGO_URI in .env — make sure password has no special chars unescaped
→ Check Atlas Network Access — your IP must be whitelisted (or use 0.0.0.0/0)

**`GROQ_API_KEY not set`**
→ Make sure .env exists in the backend/ folder (not the root)

**`pymongo.errors.ConfigurationError: ... dnspython`**
→ Run: `pip install 'pymongo[srv]'` (the [srv] extra installs dnspython for Atlas)

**Questions not loading in assessment**
→ Backend auto-regenerates them if session cache is cold — this is handled
