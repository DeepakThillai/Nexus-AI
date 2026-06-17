"use client";
import { useEffect, useState } from "react";
import { Suspense } from "react";
import { motion } from "framer-motion";
import { useSearchParams, useRouter } from "next/navigation";
import { RefreshCw, ArrowLeft, Loader2, CheckCircle, AlertTriangle, ChevronRight } from "lucide-react";
import { reroute } from "@/lib/api";
import { useStore } from "@/store/useStore";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import ParticleBackground from "@/components/ParticleBackground";
import AIThinkingOverlay from "@/components/AIThinkingOverlay";

function ConfidenceCircle({ score }: { score: number }) {
  const color = score >= 70 ? "#10B981" : score >= 40 ? "#F59E0B" : "#EF4444";
  const r = 52;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  return (
    <div className="relative w-36 h-36 mx-auto">
      <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
        <motion.circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          initial={{ strokeDasharray: `0 ${circ}` }}
          animate={{ strokeDasharray: `${dash} ${circ}` }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-black" style={{ color }}>{score}</span>
        <span className="text-white/30 text-xs">confidence</span>
      </div>
    </div>
  );
}

export default function ReroutePage() {
  return <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-[#0F1117]"><Loader2 size={28} className="animate-spin text-blue-400" /></div>}><ReroutePageInner /></Suspense>;
}

function ReroutePageInner() {
  const router  = useRouter();
  const setRole = useStore((s) => s.setRole);
  const userId  = useAuthGuard();

  const [analysis, setAnalysis] = useState<any>(null);
  const [loading,  setLoading]  = useState(true);
  const [switching, setSwitching] = useState(false);
  const [switched, setSwitched]  = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);

  useEffect(() => {
    reroute({ user_id: userId })
      .then(setAnalysis)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [userId]);

  async function handleSwitch(role: string) {
    setSwitching(true);
    try {
      const result = await reroute({ user_id: userId, new_role: role });
      setRole(role);
      setSwitched(true);
      setAnalysis(result);
    } catch(e) {
      console.error(e);
    } finally {
      setSwitching(false);
    }
  }

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-[#0F1117]">
      <Loader2 size={28} className="animate-spin text-blue-400" />
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid">
      <ParticleBackground />
      <AIThinkingOverlay 
        visible={showOverlay} 
        onComplete={() => setShowOverlay(false)} 
        duration={5000}
        subtitle="Analyzing"
        messages={[
          "Evaluating current confidence...",
          "Analyzing career deviations...",
          "Assessing alternative paths...",
          "Calculating likelihood metrics...",
          "Preparing rerouting options...",
        ]}
      />
      {!showOverlay && (
      <main className="max-w-3xl mx-auto px-6 py-10 space-y-6">
        <div className="flex items-center gap-3 mb-2">
          <button onClick={() => window.history.back()} className="btn-ghost p-2"><ArrowLeft size={16} /></button>
          <div>
            <h1 className="text-3xl font-black">Career Rerouting</h1>
            <p className="text-white/40 text-sm mt-0.5">AI-driven path adjustment based on your confidence</p>
          </div>
        </div>

        {switched && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
            className="glass border border-emerald-500/30 p-4 flex items-center gap-3">
            <CheckCircle size={18} className="text-emerald-400" />
            <div>
              <p className="font-semibold text-emerald-400">Role switched successfully!</p>
              <p className="text-white/40 text-sm">Your roadmap and market analysis have been regenerated.</p>
            </div>
          </motion.div>
        )}

        {/* Analysis result */}
        {analysis && (
          <>
            <div className="glass p-8 text-center">
              <p className="text-white/40 text-xs uppercase tracking-wider mb-4">Rerouting Assessment</p>
              <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold mb-4
                ${analysis.reroute_suggestion ? "bg-orange-500/10 text-orange-400 border border-orange-500/20"
                                              : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"}`}>
                {analysis.reroute_suggestion ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
                {analysis.reroute_suggestion ? "Reroute Suggested" : "On Track"}
              </div>
              <p className="text-white/50 text-sm leading-relaxed max-w-lg mx-auto">{analysis.reason}</p>
            </div>

            {/* Suggested roles */}
            {analysis.reroute_suggestion && analysis.suggested_roles?.length > 0 && (
              <div className="card-glow p-6">
                <h3 className="font-bold mb-4">Suggested Safer Roles</h3>
                <div className="space-y-3">
                  {analysis.suggested_roles.map((role: string) => (
                    <div key={role} className="flex items-center justify-between p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                      <span className="font-medium">{role}</span>
                      <button onClick={() => handleSwitch(role)} disabled={switching || switched}
                        className="btn-primary text-sm py-2 px-4 flex items-center gap-1.5 disabled:opacity-50">
                        {switching ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                        Switch
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Return to previous */}
            {analysis.return_previous_role_available && (
              <div className="card-glow p-6 border border-blue-500/20">
                <h3 className="font-bold mb-2 text-blue-400">Return to Previous Role</h3>
                <p className="text-white/40 text-sm mb-4">
                  Your confidence and progress indicate you may be ready to return to your previous target role.
                </p>
                <button onClick={() => handleSwitch("__return_previous__")} className="btn-primary flex items-center gap-2">
                  Return to Previous Role <ChevronRight size={15} />
                </button>
              </div>
            )}

            {!analysis.reroute_suggestion && !analysis.return_previous_role_available && (
              <div className="glass p-6 text-center">
                <CheckCircle size={32} className="text-emerald-400 mx-auto mb-3" />
                <p className="font-semibold text-emerald-400 mb-1">You're on track!</p>
                <p className="text-white/40 text-sm">Keep working through your roadmap. No rerouting needed at this time.</p>
                <button onClick={() => router.push("/roadmap")} className="btn-primary mt-4">
                  Continue Roadmap →
                </button>
              </div>
            )}
          </>
        )}
      </main>
      )}
    </div>
  );
}
