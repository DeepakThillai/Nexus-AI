/**
 * api.ts — Axios API integration layer
 * All frontend→backend communication goes through this file.
 * No business logic here — pure HTTP calls.
 */

import axios from "axios";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 120000, // 2min — LLM calls can be slow
});

// ── Request interceptor: attach user_id from localStorage ─────────
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const userId = localStorage.getItem("nexus_user_id");
    if (userId && config.data) {
      config.data = { ...config.data };
    }
  }
  return config;
});

// ── Response interceptor: unwrap errors ───────────────────────────
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.detail || err.message || "An unexpected error occurred";
    return Promise.reject(new Error(message));
  }
);

// ═══════════════════════════════════════════════════════════════════
//  TYPE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════

export interface OnboardPayload {
  name: string;
  email: string;
  target_role: string;
  skills: string[];
  strengths: string[];
  weaknesses: string[];
  experience_years: number;
  phone?: string;
}

export interface ReadinessEvaluatePayload {
  user_id: string;
  answers: string[];
}

export interface ActionQuestionsPayload {
  user_id: string;
  action_id: string;
}

export interface ActionAssessPayload {
  user_id: string;
  action_id: string;
  answers: string[];
}

export interface ReroutePayload {
  user_id: string;
  new_role?: string;
}

export interface HandsOnChatPayload {
  user_id: string;
  message: string;
  conversation_history: { role: string; content: string }[];
}

// ═══════════════════════════════════════════════════════════════════
//  API CALLS
// ═══════════════════════════════════════════════════════════════════

/** POST /api/onboard — create user + trigger market intel */
export const onboardUser = async (payload: OnboardPayload) => {
  const res = await api.post("/api/onboard", payload);
  return res.data;
};

/** POST /api/readiness/start — generate 10 questions */
export const startReadiness = async (user_id: string) => {
  const res = await api.post("/api/readiness/start", { user_id });
  return res.data as { user_id: string; questions: string[] };
};

/** POST /api/readiness/evaluate — submit answers, get score */
export const evaluateReadiness = async (payload: ReadinessEvaluatePayload) => {
  const res = await api.post("/api/readiness/evaluate", payload);
  return res.data;
};

/** GET /api/dashboard/:user_id */
export const getDashboard = async (user_id: string) => {
  const res = await api.get(`/api/dashboard/${user_id}`);
  return res.data;
};

/** GET /api/roadmap/:user_id */
export const getRoadmap = async (user_id: string) => {
  const res = await api.get(`/api/roadmap/${user_id}`);
  return res.data;
};

/** POST /api/roadmap/regenerate */
export const regenerateRoadmap = async (user_id: string, target_role?: string) => {
  const res = await api.post("/api/roadmap/regenerate", { user_id, target_role });
  return res.data;
};

/** POST /api/action/questions — get 10 questions for an action */
export const getActionQuestions = async (payload: ActionQuestionsPayload) => {
  const res = await api.post("/api/action/questions", payload);
  return res.data as { action_id: string; action_title: string; questions: string[] };
};

/** POST /api/action/assess — submit answers, update score + confidence */
export const assessAction = async (payload: ActionAssessPayload) => {
  const res = await api.post("/api/action/assess", payload);
  return res.data;
};

/** GET /api/market/:user_id */
export const getMarket = async (user_id: string) => {
  const res = await api.get(`/api/market/${user_id}`);
  return res.data;
};

/** POST /api/reroute */
export const reroute = async (payload: ReroutePayload) => {
  const res = await api.post("/api/reroute", payload);
  return res.data;
};

/** POST /api/feedback */
export const generateFeedback = async (user_id: string) => {
  const res = await api.post("/api/feedback", { user_id });
  return res.data;
};

/** POST /api/hands-on/chat */
export const handsOnChat = async (payload: HandsOnChatPayload) => {
  const res = await api.post("/api/hands-on/chat", payload);
  return res.data as { reply: string; conversation_history: { role: string; content: string }[] };
};