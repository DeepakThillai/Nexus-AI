"use client";
import { useEffect, useState } from "react";
import { Suspense } from "react";
import { motion } from "framer-motion";
import { useSearchParams } from "next/navigation";import { TrendingUp, AlertCircle, Lightbulb, Target, Calendar, Loader2, ArrowLeft, RefreshCw, Smile } from "lucide-react";
import { generateFeedback } from "@/lib/api";
import { useStore } from "@/store/useStore";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import ParticleBackground from "@/components/ParticleBackground";

function InsightCard({ insight }: { insight: any }) {
  return (
    <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06] space-y-2">
      <p className="font-semibold text-sm flex items-center gap-2">
        <Lightbulb size={13} className="text-yellow-400" /> {insight.insight}
      </p>
      <p className="text-white/40 text-xs"><span className="text-white/20">Evidence:</span> {insight.evidence}</p>
      <p className="text-blue-300 text-xs"><span className="text-white/20">Recommendation:</span> {insight.recommendation}</p>
    </div>
  );
}

function AdjustmentCard({ adj }: { adj: any }) {
  return (
    <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
      <div className="flex items-center gap-2 mb-1">
        <span className="tag">{adj.adjustment_type}</span>
      </div>
      <p className="text-sm text-white/70 mb-1">{adj.specific_change}</p>
      <p className="text-white/30 text-xs">{adj.reason}</p>
    </div>
  );
}

export default function FeedbackPage() {
  return <Suspense fallback={<div className="min-h-screen flex flex-col items-center justify-center bg-[#0F1117] gap-4"><Loader2 size={28} className="animate-spin text-blue-400" /></div>}><FeedbackPageInner /></Suspense>;
}

function FeedbackPageInner() {
  const userId = useAuthGuard();

  const [feedback, setFeedback] = useState<any>(null);
  const [loading,  setLoading]  = useState(false);

  async function fetchFeedback() {
    setLoading(true);
    try {
      const res = await generateFeedback(userId);
      setFeedback(res.feedback_analysis);
    } catch(e) { console.error(e); }
    setLoading(false);
  }

  useEffect(() => { fetchFeedback(); }, [userId]);

  if (loading) return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0F1117] gap-4">
      <Loader2 size={28} className="animate-spin text-blue-400" />
      <p className="text-white/40">Generating comprehensive feedback via Agent...</p>
    </div>
  );

  const f = feedback;

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid">
      <ParticleBackground />
      <main className="max-w-5xl mx-auto px-6 py-10 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <button onClick={() => window.history.back()} className="btn-ghost p-2"><ArrowLeft size={16} /></button>
            <div>
              <h1 className="text-3xl font-black">AI Feedback Report</h1>
              <p className="text-white/40 text-sm mt-0.5">Comprehensive progress analysis</p>
            </div>
          </div>
          <button onClick={fetchFeedback} className="btn-ghost flex items-center gap-2 text-sm">
            <RefreshCw size={13} /> Regenerate
          </button>
        </div>

        {!f && !loading && (
          <div className="glass p-12 text-center">
            <TrendingUp size={32} className="text-blue-400 mx-auto mb-4" />
            <p className="text-white/50 mb-4">No feedback generated yet.</p>
            <button onClick={fetchFeedback} className="btn-primary">Generate Feedback</button>
          </div>
        )}

        {f && (
          <>
            {/* Encouragement banner */}
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="glass p-6 border border-blue-500/20 bg-blue-500/5">
              <div className="flex items-start gap-3">
                <Smile size={22} className="text-blue-400 shrink-0 mt-0.5" />
                <p className="text-white/70 italic leading-relaxed">{f.encouragement_message}</p>
              </div>
            </motion.div>

            {/* Key metrics */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: "Overall Rating",   value: f.overall_progress_rating, color: "text-emerald-400" },
                { label: "Velocity",         value: f.velocity_assessment,     color: "text-blue-400" },
                { label: "Motivation",       value: f.motivation_level,        color: "text-violet-400" },
                { label: "Deviation Risk",   value: f.updated_deviation_risk,  color: "text-yellow-400" },
              ].map(({ label, value, color }) => (
                <div key={label} className="card-glow p-5">
                  <p className="text-white/40 text-xs uppercase tracking-wider mb-2">{label}</p>
                  <p className={`text-xl font-black ${color} capitalize`}>{value || "N/A"}</p>
                </div>
              ))}
            </div>

            {/* Progress + confidence */}
            <div className="grid lg:grid-cols-2 gap-5">
              <div className="card-glow p-5">
                <h3 className="font-bold text-sm text-white/60 mb-3 uppercase tracking-wider">
                  <TrendingUp size={13} className="inline mr-1.5 text-blue-400" />Strengths Observed
                </h3>
                <div className="space-y-2">
                  {f.strengths_observed?.length > 0
                    ? f.strengths_observed.map((s: string, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-sm text-white/60">
                          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" /> {s}
                        </div>
                      ))
                    : <p className="text-white/30 text-sm">Complete actions to see strengths</p>
                  }
                </div>
              </div>
              <div className="card-glow p-5">
                <h3 className="font-bold text-sm text-white/60 mb-3 uppercase tracking-wider">
                  <AlertCircle size={13} className="inline mr-1.5 text-orange-400" />Areas of Concern
                </h3>
                <div className="space-y-2">
                  {f.areas_of_concern?.length > 0
                    ? f.areas_of_concern.map((s: string, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-sm text-white/60">
                          <div className="w-1.5 h-1.5 rounded-full bg-orange-400 shrink-0" /> {s}
                        </div>
                      ))
                    : <p className="text-white/30 text-sm">No concerns identified yet</p>
                  }
                </div>
              </div>
            </div>

            {/* Learning insights */}
            {f.learning_insights?.length > 0 && (
              <div className="card-glow p-5">
                <h3 className="font-bold text-sm text-white/60 mb-4 uppercase tracking-wider">Learning Insights</h3>
                <div className="grid md:grid-cols-2 gap-3">
                  {f.learning_insights.map((ins: any, i: number) => <InsightCard key={i} insight={ins} />)}
                </div>
              </div>
            )}

            {/* Recommended adjustments */}
            {f.recommended_adjustments?.length > 0 && (
              <div className="card-glow p-5">
                <h3 className="font-bold text-sm text-white/60 mb-4 uppercase tracking-wider">Recommended Adjustments</h3>
                <div className="grid md:grid-cols-2 gap-3">
                  {f.recommended_adjustments.map((adj: any, i: number) => <AdjustmentCard key={i} adj={adj} />)}
                </div>
              </div>
            )}

            {/* Next checkpoint */}
            {f.next_checkpoint_date && (
              <div className="card-glow p-4 flex items-center gap-3">
                <Calendar size={18} className="text-blue-400" />
                <div>
                  <p className="font-semibold text-sm">Next Checkpoint</p>
                  <p className="text-white/40 text-sm">{f.next_checkpoint_date}</p>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
