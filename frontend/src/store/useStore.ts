/**
 * store/useStore.ts — Zustand global state
 * Stores UI state only. All authoritative data lives in MongoDB.
 * Uses sessionStorage — clears automatically when browser tab is closed.
 */
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

interface UserState {
  userId: string | null;
  userName: string | null;
  targetRole: string | null;
  confidenceScore: number;
  readinessScore: number;
  readinessStatus: string;
  isOnboarded: boolean;
  isReadinessDone: boolean;
  // Actions
  setUser: (id: string, name: string) => void;
  setRole: (role: string) => void;
  setConfidence: (score: number) => void;
  setReadiness: (score: number, status: string) => void;
  markOnboarded: () => void;
  markReadinessDone: () => void;
  reset: () => void;
}

export const useStore = create<UserState>()(
  persist(
    (set) => ({
      userId: null,
      userName: null,
      targetRole: null,
      confidenceScore: 0,
      readinessScore: 0,
      readinessStatus: "",
      isOnboarded: false,
      isReadinessDone: false,

      setUser: (id, name) => set({ userId: id, userName: name }),
      setRole: (role) => set({ targetRole: role }),
      setConfidence: (score) => set({ confidenceScore: score }),
      setReadiness: (score, status) =>
        set({ readinessScore: score, readinessStatus: status }),
      markOnboarded: () => set({ isOnboarded: true }),
      markReadinessDone: () => set({ isReadinessDone: true }),
      reset: () =>
        set({
          userId: null,
          userName: null,
          targetRole: null,
          confidenceScore: 0,
          readinessScore: 0,
          readinessStatus: "",
          isOnboarded: false,
          isReadinessDone: false,
        }),
    }),
    {
      name: "nexus-user-store",
      storage: createJSONStorage(() => sessionStorage),
    }
  )
);
