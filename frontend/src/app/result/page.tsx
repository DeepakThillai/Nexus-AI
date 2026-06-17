"use client";
import { useEffect, useState } from "react";
import { Suspense } from "react";
import { motion } from "framer-motion";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowRight, TrendingDown, TrendingUp, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { getDashboard } from "@/lib/api";
import { useStore } from "@/store/useStore";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import ParticleBackground from "@/components/ParticleBackground";

const statusConfig = {
  underqualified: { color: "text-orange-400", bg: "bg-orange-500/10", border: "border-orange-500/20", icon: TrendingDown, label: "Underqualified" },
  qualified:      { color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20", icon: CheckCircle, label: "Qualified" },
  overqualified:  { color: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/20", icon: TrendingUp, label: "Overqualified" },
};

function ScoreGauge({ score }: { score: number }) {
  const angle    = -135 + (score / 100) * 270;
  const color    = score >= 70 ? "#10B981" : score >= 40 ? "#F59E0B" : "#EF4444";
  const circumf  = 2 * Math.PI * 70;
  const dash     = (score / 100) * circumf * 0.75;

  return (
    <div className="relative w-52 h-52 mx-auto">
      <svg viewBox="0 0 180 180" className="w-full h-full -rotate-[135deg]">
        <circle cx="90" cy="90" r="70" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${circumf * 0.75} ${circumf}`} />
        <motion.circle cx="90" cy="90" r="70" fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${dash} ${circumf}`}
          initial={{ strokeDasharray: `0 ${circumf}` }}
          animate={{ strokeDasharray: `${dash} ${circumf}` }}
          transition={{ duration: 1.5, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className="text-5xl font-black"
          style={{ color }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          {score}
        </motion.span>
        <span className="text-white/40 text-sm">out of 100</span>
      </div>
    </div>
  );
}

export default function ResultPage() {
  return <Suspense fallback={<div className="min-h-screen flex items-center justify-center"><Loader2 size={32} className="animate-spin text-blue-400" /></div>}><ResultPageInner /></Suspense>;
}

function ResultPageInner() {
  const router = useRouter();
  const userId = useAuthGuard();

  const [data,    setData]    = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!userId) return;
    getDashboard(userId)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [userId]);

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-blue-400" />
    </div>
  );

  if (!data) return <div className="min-h-screen flex items-center justify-center text-white/40">No data found</div>;

  const readiness = data.readiness;
  const status    = (readiness.status || "underqualified") as keyof typeof statusConfig;
  const cfg       = statusConfig[status] || statusConfig.underqualified;
  const StatusIcon = cfg.icon;

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid flex items-center justify-center px-4 py-12">
      <ParticleBackground />
      <div className="fixed top-1/4 right-1/4 w-80 h-80 bg-violet-600/8 rounded-full blur-3xl pointer-events-none" />

      <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-2xl space-y-5">
        <div className="text-center">
          <h1 className="text-3xl font-black mb-1">Assessment Complete</h1>
          <p className="text-white/40">Your AI-powered readiness evaluation is ready.</p>
        </div>

        {/* Score card */}
        <div className="glass p-8 text-center">
          <ScoreGauge score={readiness.score} />

          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full mt-4 ${cfg.bg} ${cfg.border} border`}>
            <StatusIcon size={16} className={cfg.color} />
            <span className={`font-bold ${cfg.color}`}>{cfg.label}</span>
          </div>

          <p className="text-white/50 text-sm leading-relaxed mt-4 max-w-lg mx-auto">
            {readiness.evaluation_summary}
          </p>
        </div>

        {/* Adjacent roles */}
        <div className="grid grid-cols-2 gap-4">
          <div className="glass p-5">
            <h3 className="font-semibold text-sm text-orange-400 mb-3 flex items-center gap-2">
              <TrendingDown size={14} /> Safer Adjacent Roles
            </h3>
            <div className="space-y-2">
              {readiness.safer_adjacent_roles?.map((r: string) => (
                <div key={r} className="text-sm text-white/60 py-1.5 px-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">{r}</div>
              ))}
            </div>
          </div>
          <div className="glass p-5">
            <h3 className="font-semibold text-sm text-blue-400 mb-3 flex items-center gap-2">
              <TrendingUp size={14} /> Advanced Roles
            </h3>
            <div className="space-y-2">
              {readiness.advanced_adjacent_roles?.map((r: string) => (
                <div key={r} className="text-sm text-white/60 py-1.5 px-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">{r}</div>
              ))}
            </div>
          </div>
        </div>

        <button onClick={() => router.push("/dashboard")}
          className="btn-primary w-full flex items-center justify-center gap-2 py-4 text-base">
          View My Dashboard <ArrowRight size={18} />
        </button>
      </motion.div>
    </div>
  );
}
