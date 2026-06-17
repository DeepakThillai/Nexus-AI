"use client";
import { useEffect, useState } from "react";
import { Suspense } from "react";
import { motion } from "framer-motion";
import { useSearchParams } from "next/navigation";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { TrendingUp, TrendingDown, Minus, Building2, Loader2, ArrowLeft, Zap } from "lucide-react";
import { getMarket } from "@/lib/api";
import { useStore } from "@/store/useStore";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import ParticleBackground from "@/components/ParticleBackground";
import AIThinkingOverlay from "@/components/AIThinkingOverlay";

function MetricBadge({ label, value, type }: { label: string; value: string; type?: string }) {
  const colors: any = {
    high:     "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    medium:   "text-yellow-400  bg-yellow-500/10  border-yellow-500/20",
    low:      "text-blue-400    bg-blue-500/10    border-blue-500/20",
    growing:  "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    stable:   "text-blue-400    bg-blue-500/10    border-blue-500/20",
    declining:"text-red-400     bg-red-500/10     border-red-500/20",
    abundant: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    moderate: "text-yellow-400  bg-yellow-500/10  border-yellow-500/20",
    scarce:   "text-red-400     bg-red-500/10     border-red-500/20",
  };
  const cls = colors[value?.toLowerCase()] || "text-white/60 bg-white/[0.06] border-white/[0.08]";
  return (
    <div className={`px-3 py-1.5 rounded-lg border text-xs font-semibold ${cls}`}>
      <span className="text-white/30 mr-1.5">{label}:</span>{value}
    </div>
  );
}

export default function MarketPage() {
  return <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-[#0F1117]"><Loader2 size={28} className="animate-spin text-blue-400" /></div>}><MarketPageInner /></Suspense>;
}

function MarketPageInner() {
  const userId = useAuthGuard();

  const [market,  setMarket]  = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showOverlay, setShowOverlay] = useState(true);

  useEffect(() => {
    getMarket(userId).then(d => { setMarket(d.market_analysis); setLoading(false); }).catch(console.error);
  }, [userId]);

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-[#0F1117]">
      <Loader2 size={28} className="animate-spin text-blue-400" />
    </div>
  );

  if (!market) return <div className="min-h-screen flex items-center justify-center text-white/40">No market data</div>;

  const demandData   = [{ name: "Demand", score: market.demand_score }];
  const adjacentData = market.adjacent_safer_roles?.map((r: any) => ({
    name: r.role.substring(0, 16),
    demand: r.demand_score,
  })) || [];

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid">
      <ParticleBackground />
      <AIThinkingOverlay 
        visible={showOverlay} 
        onComplete={() => setShowOverlay(false)} 
        duration={5000}
        subtitle="Searching"
        messages={[
          "Analyzing LinkedIn trends...",
          "Searching Naukri data...",
          "Evaluating market competition...",
          "Gathering salary insights...",
          "Refining AI recommendations...",
        ]}
      />
      {!showOverlay && (
      <main className="max-w-5xl mx-auto px-6 py-10 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <button onClick={() => window.history.back()} className="btn-ghost p-2"><ArrowLeft size={16} /></button>
          <div>
            <h1 className="text-3xl font-black">Market Intelligence</h1>
            <p className="text-white/40 text-sm mt-0.5">{market.role_title} · {market.last_updated}</p>
          </div>
        </div>

        {/* Key metrics */}
        <div className="flex flex-wrap gap-2">
          <MetricBadge label="Competition" value={market.competition_level} />
          <MetricBadge label="Entry Barrier" value={market.entry_barrier} />
          <MetricBadge label="Trend" value={market.market_trend} />
          <MetricBadge label="Availability" value={market.job_availability} />
          <MetricBadge label="Saturation" value={market.market_saturation} />
        </div>

        {/* Salary + Demand */}
        <div className="grid grid-cols-3 gap-4">
          <div className="card-glow p-5 col-span-1">
            <p className="text-white/40 text-xs uppercase tracking-wider mb-2">Avg Salary (USD)</p>
            <p className="text-3xl font-black gradient-text">{market.avg_salary_range_usd}</p>
          </div>
          <div className="card-glow p-5 col-span-1">
            <p className="text-white/40 text-xs uppercase tracking-wider mb-2">Experience Required</p>
            <p className="text-3xl font-black text-violet-400">{market.required_experience_years} yrs</p>
          </div>
          <div className="card-glow p-5 col-span-1">
            <p className="text-white/40 text-xs uppercase tracking-wider mb-2">Demand Score</p>
            <div className="flex items-end gap-2">
              <p className="text-3xl font-black text-blue-400">{market.demand_score}</p>
              <span className="text-white/30 text-sm mb-1">/100</span>
            </div>
            <div className="h-1.5 bg-white/[0.06] rounded mt-3">
              <div className="h-full bg-blue-500 rounded" style={{ width: `${market.demand_score}%` }} />
            </div>
          </div>
        </div>

        {/* Charts + lists */}
        <div className="grid lg:grid-cols-2 gap-5">
          {/* In-demand skills */}
          <div className="card-glow p-5">
            <h3 className="font-bold text-sm text-white/60 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Zap size={13} className="text-blue-400" /> In-Demand Skills
            </h3>
            <div className="flex flex-wrap gap-2">
              {market.in_demand_skills?.map((s: string) => (
                <span key={s} className="tag">{s}</span>
              ))}
            </div>
          </div>

          {/* Key hiring companies */}
          <div className="card-glow p-5">
            <h3 className="font-bold text-sm text-white/60 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Building2 size={13} className="text-violet-400" /> Key Hiring Companies
            </h3>
            <div className="space-y-2">
              {market.key_hiring_companies?.map((c: string) => (
                <div key={c} className="flex items-center gap-2 py-1.5 px-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
                  <div className="w-2 h-2 rounded-full bg-violet-500" />
                  <span className="text-sm text-white/70">{c}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Adjacent roles chart */}
        {adjacentData.length > 0 && (
          <div className="card-glow p-5">
            <h3 className="font-bold text-sm text-white/60 uppercase tracking-wider mb-4">Adjacent Safer Roles — Demand Score</h3>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={adjacentData} barSize={28}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="name" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#1A1D2E", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, color: "#fff" }} />
                <Bar dataKey="demand" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Market notes */}
        {market.market_notes && (
          <div className="card-glow p-5">
            <h3 className="font-bold text-sm text-white/60 uppercase tracking-wider mb-3">Market Intelligence Notes</h3>
            <p className="text-white/60 text-sm leading-relaxed">{market.market_notes}</p>
          </div>
        )}
      </main>
      )}
    </div>
  );
}
