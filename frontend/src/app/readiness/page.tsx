"use client";
import { useState, useEffect } from "react";
import { Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter, useSearchParams } from "next/navigation";
import { Brain, ChevronRight, Loader2, CheckCircle } from "lucide-react";
import Image from "next/image";
import { startReadiness, evaluateReadiness } from "@/lib/api";
import { useStore } from "@/store/useStore";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import ParticleBackground from "@/components/ParticleBackground";

export default function ReadinessPage() {
  return <Suspense fallback={<div className="min-h-screen bg-[#0F1117] flex items-center justify-center"><Loader2 size={32} className="animate-spin text-blue-400" /></div>}><ReadinessPageInner /></Suspense>;
}

function ReadinessPageInner() {
  const router         = useRouter();
  const userId         = useAuthGuard();
  const setReadiness   = useStore((s) => s.setReadiness);
  const setConfidence  = useStore((s) => s.setConfidence);
  const markReadiness  = useStore((s) => s.markReadinessDone);

  const [phase,     setPhase]     = useState<"loading" | "questions" | "submitting" | "done">("loading");
  const [questions, setQuestions] = useState<string[]>([]);
  const [answers,   setAnswers]   = useState<string[]>([]);
  const [current,   setCurrent]   = useState(0);
  const [answer,    setAnswer]    = useState("");
  const [error,     setError]     = useState("");

  useEffect(() => {
    if (!userId) return;
    startReadiness(userId)
      .then((data) => {
        setQuestions(data.questions);
        setAnswers(new Array(data.questions.length).fill(""));
        setPhase("questions");
      })
      .catch((e) => setError(e.message));
  }, [userId]);

  function handleNext() {
    const updated = [...answers];
    updated[current] = answer;
    setAnswers(updated);
    setAnswer(updated[current + 1] || "");
    setCurrent((c) => c + 1);
  }

  function handleBack() {
    const updated = [...answers];
    updated[current] = answer;
    setAnswers(updated);
    setAnswer(updated[current - 1] || "");
    setCurrent((c) => c - 1);
  }

  async function handleSubmit() {
    const finalAnswers = [...answers];
    finalAnswers[current] = answer;
    setPhase("submitting");
    try {
      const result = await evaluateReadiness({ user_id: userId, answers: finalAnswers });
      setReadiness(result.score, result.status);
      setConfidence(result.confidence_score);
      markReadiness();
      router.push(`/result`);
    } catch (e: any) {
      setError(e.message);
      setPhase("questions");
    }
  }

  const progress = questions.length > 0 ? ((current + 1) / questions.length) * 100 : 0;
  const isLast   = current === questions.length - 1;

  if (error) return (
    <div className="min-h-screen flex items-center justify-center text-red-400">
      <div className="glass p-8 max-w-md text-center">
        <p className="font-semibold mb-4">Error: {error}</p>
        <button onClick={() => router.push("/dashboard")} className="btn-ghost">Go to Dashboard</button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid flex items-center justify-center px-4">
      <ParticleBackground />
      <div className="fixed top-1/3 left-1/4 w-72 h-72 bg-blue-600/8 rounded-full blur-3xl pointer-events-none" />

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mx-auto mb-4">
            <Image src="/nexus_logo.png" alt="Nexus-AI" width={64} height={64} className="object-contain" />
          </div>
          <h1 className="text-3xl font-black mb-1">Readiness Assessment</h1>
          <p className="text-white/40">Agent-powered evaluation • 10 questions • ~5 minutes</p>
        </div>

        {phase === "loading" && (
          <div className="glass p-12 text-center">
            <Loader2 size={32} className="animate-spin text-blue-400 mx-auto mb-4" />
            <p className="text-white/50">Generating personalised questions via Agent...</p>
          </div>
        )}

        {phase === "questions" && questions.length > 0 && (
          <div className="glass p-8">
            {/* Progress bar */}
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-white/40">Question {current + 1} of {questions.length}</span>
              <span className="text-sm font-semibold text-blue-400">{Math.round(progress)}%</span>
            </div>
            <div className="h-1.5 bg-white/[0.06] rounded-full mb-8 overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 to-violet-500 rounded-full"
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={current}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <p className="text-xl font-semibold mb-6 leading-relaxed">{questions[current]}</p>
                <textarea
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Type your answer here..."
                  rows={5}
                  className="input-field resize-none mb-6"
                />
              </motion.div>
            </AnimatePresence>

            <div className="flex justify-between">
              <button onClick={handleBack} disabled={current === 0} className="btn-ghost disabled:opacity-0">
                ← Back
              </button>
              {isLast ? (
                <button onClick={handleSubmit} disabled={!answer.trim()} className="btn-primary flex items-center gap-2">
                  <CheckCircle size={16} /> Submit All Answers
                </button>
              ) : (
                <button onClick={handleNext} disabled={!answer.trim()} className="btn-primary flex items-center gap-2">
                  Next <ChevronRight size={16} />
                </button>
              )}
            </div>
          </div>
        )}

        {phase === "submitting" && (
          <div className="glass p-12 text-center">
            <Loader2 size={32} className="animate-spin text-violet-400 mx-auto mb-4" />
            <p className="text-white/60 font-semibold">Evaluating your responses...</p>
            <p className="text-white/30 text-sm mt-2">Agent is analysing all 10 answers</p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
