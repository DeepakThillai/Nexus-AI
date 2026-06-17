"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { Brain, Mail, Lock, ArrowRight, Loader2, RefreshCw, Eye, EyeOff } from "lucide-react";
import Image from "next/image";
import { api } from "@/lib/api";
import { useStore } from "@/store/useStore";
import ParticleBackground from "@/components/ParticleBackground";

type Step = "email" | "password";

export default function AuthPage() {
  const router  = useRouter();
  const setUser = useStore((s) => s.setUser);

  const [step,        setStep]        = useState<Step>("email");
  const [email,       setEmail]       = useState("");
  const [password,    setPassword]    = useState("");
  const [showPw,      setShowPw]      = useState(false);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState("");
  const [info,        setInfo]        = useState("");   // success / info messages
  const [userId,      setUserId]      = useState("");
  const [isNewUser,   setIsNewUser]   = useState(false);
  const [forgotSent,  setForgotSent]  = useState(false);

  // ── Step 1: submit email ────────────────────────────────────────
  async function handleEmailSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim()) return;
    setLoading(true);
    setError("");
    setInfo("");
    try {
      const res = await api.post("/api/auth/login", { email: email.toLowerCase().trim() });
      const d   = res.data;

      setUserId(d.user_id || "");
      setIsNewUser(!!d.is_new_user);

      if (d.is_new_user) {
        setInfo("Account created! Check your email for your password.");
      }
      setStep("password");
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  // ── Step 2: verify password ─────────────────────────────────────
  async function handlePasswordSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!password.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.post("/api/auth/verify", {
        email: email.toLowerCase().trim(),
        password,
      });
      const d = res.data;

      setUser(d.user_id, "");

      if (d.is_new_user) {
        router.push(`/onboarding?uid=${d.user_id}&email=${encodeURIComponent(email.toLowerCase().trim())}`);
      } else {
        router.push(`/dashboard`);
      }
    } catch (err: any) {
      setError(err.message || "Incorrect password");
    } finally {
      setLoading(false);
    }
  }

  // ── Forgot password ─────────────────────────────────────────────
  async function handleForgotPassword() {
    if (!email.trim()) return;
    setLoading(true);
    setError("");
    try {
      await api.post("/api/auth/forgot-password", { email: email.toLowerCase().trim() });
      setForgotSent(true);
      setInfo("A new password has been sent to your email.");
    } catch (err: any) {
      setError(err.message || "Failed to send password reset");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid flex items-center justify-center px-4">
      <ParticleBackground />
      <div className="fixed top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-blue-600/8 rounded-full blur-3xl pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md"
      >
        {/* Logo */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center mx-auto mb-4">
            <Image src="/nexus_logo.png" alt="Nexus-AI" width={80} height={80} className="object-contain drop-shadow-glow" />
          </div>
          <h1 className="text-3xl font-black mb-2">Welcome to Nexus-AI</h1>
          <p className="text-white/40 text-sm">
            {step === "email"
              ? "Enter your email to get started or continue your journey."
              : isNewUser
                ? "Your account is ready. Check your email for the password."
                : `Signing in as ${email}`}
          </p>
        </div>

        <div className="glass p-8 space-y-5">
          <AnimatePresence mode="wait">

            {/* ── Email step ── */}
            {step === "email" && (
              <motion.form
                key="email-step"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                onSubmit={handleEmailSubmit}
                className="space-y-5"
              >
                <div>
                  <label className="block text-sm font-medium text-white/60 mb-2">Email address</label>
                  <div className="relative">
                    <Mail size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@example.com"
                      className="input-field pl-10"
                      required
                      autoFocus
                    />
                  </div>
                </div>

                {error && (
                  <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-xl p-3">{error}</div>
                )}

                <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
                  {loading ? <><Loader2 size={16} className="animate-spin" /> Checking...</> : <>Continue <ArrowRight size={16} /></>}
                </button>
              </motion.form>
            )}

            {/* ── Password step ── */}
            {step === "password" && (
              <motion.form
                key="password-step"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                onSubmit={handlePasswordSubmit}
                className="space-y-5"
              >
                {/* Email display (read-only) */}
                <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.08]">
                  <Mail size={14} className="text-white/30 shrink-0" />
                  <span className="text-sm text-white/60 flex-1 truncate">{email}</span>
                  <button
                    type="button"
                    onClick={() => { setStep("email"); setError(""); setInfo(""); setPassword(""); }}
                    className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                  >
                    Change
                  </button>
                </div>

                {/* Info banner */}
                {info && (
                  <div className="text-emerald-400 text-sm bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3">
                    {info}
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-white/60 mb-2">Password</label>
                  <div className="relative">
                    <Lock size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30" />
                    <input
                      type={showPw ? "text" : "password"}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="Enter your password"
                      className="input-field pl-10 pr-10"
                      required
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => setShowPw(!showPw)}
                      className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
                    >
                      {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
                    </button>
                  </div>
                </div>

                {error && (
                  <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-xl p-3">{error}</div>
                )}

                <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
                  {loading ? <><Loader2 size={16} className="animate-spin" /> Signing in...</> : <>Sign In <ArrowRight size={16} /></>}
                </button>

                {/* Forgot password */}
                <div className="text-center pt-1">
                  {forgotSent ? (
                    <p className="text-emerald-400 text-xs">New password sent to your email.</p>
                  ) : (
                    <button
                      type="button"
                      onClick={handleForgotPassword}
                      disabled={loading}
                      className="text-white/30 hover:text-blue-400 text-xs transition-colors flex items-center gap-1.5 mx-auto"
                    >
                      <RefreshCw size={12} /> Forgot password? Send a new one
                    </button>
                  )}
                </div>
              </motion.form>
            )}
          </AnimatePresence>
        </div>

        <p className="text-center text-white/20 text-xs mt-6">
          New email → account created, password sent · Existing email → sign in
        </p>
      </motion.div>
    </div>
  );
}
