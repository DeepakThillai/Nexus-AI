"use client";
import { useEffect, useState } from "react";
import { Suspense } from "react";
import { motion } from "framer-motion";
import { useRouter, useSearchParams } from "next/navigation";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Brain, TrendingUp, Target, Calendar, Map, BarChart3, MessageSquare, RefreshCw, Loader2, AlertTriangle, HelpCircle, Users } from "lucide-react";
import Image from "next/image";
import { getDashboard } from "@/lib/api";
import Link from "next/link";
import { useStore } from "@/store/useStore";
import ParticleBackground from "@/components/ParticleBackground";

function StatCard({ title, value, subtitle, icon: Icon, color, glow }: any) {
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="card-glow p-6">
      <div className="flex items-start justify-between mb-4">
        <div className={`w-11 h-11 rounded-xl ${glow} flex items-center justify-center`}>
          <Icon size={20} className={color} />
        </div>
      </div>
      <div className="text-3xl font-black mb-1">{value}</div>
      <div className="text-white/50 text-sm font-medium">{title}</div>
      {subtitle && <div className="text-white/30 text-xs mt-0.5">{subtitle}</div>}
    </motion.div>
  );
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass px-3 py-2 text-xs">
      <p className="text-white/60 mb-1">{label}</p>
      <p className="font-bold text-blue-400">{payload[0].value}/100</p>
    </div>
  );
};

export default function DashboardPage() {
  return <Suspense fallback={<div className="min-h-screen bg-[#0F1117] flex items-center justify-center"><Loader2 size={32} className="animate-spin text-blue-400" /></div>}><DashboardPageInner /></Suspense>;
}

function DashboardPageInner() {
  const router   = useRouter();
  const params   = useSearchParams();
  const storeUid = useStore((s) => s.userId);
  const userId   = params.get("uid") || storeUid || "";

  const [data,    setData]    = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (!userId) { router.push("/auth"); return; }
    setLoading(true);
    getDashboard(userId)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [userId, refresh]);

  if (loading) return (
    <div className="min-h-screen bg-[#0F1117] flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-blue-400" />
    </div>
  );

  if (!data) return <div className="min-h-screen flex items-center justify-center text-white/40">No data</div>;

  const { profile, confidence_score, readiness, progress, roadmap_summary, reroute_state, career_state, weekly_scores } = data;

  const navLinks = [
    { href: `/roadmap?uid=${userId}`,    icon: Map,          label: "Roadmap" },
    { href: `/market?uid=${userId}`,     icon: BarChart3,    label: "Market" },
    { href: `/hands-on?uid=${userId}`,   icon: MessageSquare,label: "Hands-On" },
    { href: `/feedback?uid=${userId}`,   icon: TrendingUp,   label: "Feedback" },
    { href: `/reroute?uid=${userId}`,    icon: RefreshCw,    label: "Reroute" },
    { href: `/help`,                     icon: HelpCircle,   label: "Help" },
    { href: `/credits`,                  icon: Users,        label: "Credits" },
  ];

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid">
      <ParticleBackground />
      <div className="fixed top-0 left-0 w-80 h-80 bg-blue-600/5 rounded-full blur-3xl pointer-events-none" />

      {/* Top nav */}
      <nav className="glass border-b border-white/[0.06] sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center">
              <Image src="/nexus_logo.png" alt="Nexus-AI" width={32} height={32} className="object-contain" />
            </div>
            <span className="font-bold">Nexus-AI</span>
            <span className="text-white/20 text-sm hidden md:block">· {career_state?.current_target_role || profile?.target_role}</span>
          </div>
          <div className="flex items-center gap-2">
            {navLinks.map(({ href, icon: Icon, label }) => (
              <Link key={href} href={href} className="btn-ghost text-xs py-2 px-3 flex items-center gap-1.5">
                <Icon size={13} />{label}
              </Link>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Welcome */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-black">
              Welcome back, <span className="gradient-text">{profile?.name || "Navigator"}</span>
            </h1>
            <p className="text-white/40 mt-1">
              Targeting: <span className="text-white/70 font-medium">{profile?.target_role}</span>
            </p>
          </div>
          <button onClick={() => setRefresh(r => r + 1)} className="btn-ghost flex items-center gap-2 text-sm">
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {/* Reroute alert */}
        {confidence_score < 40 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="glass border border-orange-500/30 p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertTriangle size={18} className="text-orange-400" />
              <div>
                <p className="font-semibold text-orange-400 text-sm">Confidence is low ({confidence_score})</p>
                <p className="text-white/40 text-xs">Consider reviewing rerouting options</p>
              </div>
            </div>
            <Link href={`/reroute?uid=${userId}`} className="btn-ghost text-sm py-2 px-4 border-orange-500/30">
              View Options →
            </Link>
          </motion.div>
        )}

        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {(() => {
            const skillLevel = confidence_score >= 80 ? "HIGH" : confidence_score >= 30 ? "MEDIUM" : "LOW";
            const skillColor = confidence_score >= 80 ? "text-emerald-400" : confidence_score >= 30 ? "text-yellow-400" : "text-red-400";
            const skillGlow = confidence_score >= 80 ? "bg-emerald-500/15" : confidence_score >= 30 ? "bg-yellow-500/15" : "bg-red-500/15";
            return <StatCard title="Skill Level" value={skillLevel} subtitle={`${confidence_score}% confidence`} icon={Brain} color={skillColor} glow={skillGlow} />;
          })()}
          <StatCard title="Confidence Score" value={confidence_score} subtitle="Updates per action"
            icon={TrendingUp} color="text-violet-400" glow="bg-violet-500/15" />
          <StatCard title="Progress" value={`${roadmap_summary?.completion_pct ?? 0}%`}
            subtitle={`${roadmap_summary?.done_actions}/${roadmap_summary?.total_actions} actions`}
            icon={Target} color="text-emerald-400" glow="bg-emerald-500/15" />
          <StatCard title="Weeks Done" value={progress?.weeks_completed ?? 0} subtitle="of 20 total weeks"
            icon={Calendar} color="text-yellow-400" glow="bg-yellow-500/15" />
        </div>

        {/* Chart + quick actions */}
        <div className="grid lg:grid-cols-3 gap-5">
          {/* Score line chart */}
          <div className="card-glow p-6 lg:col-span-2">
            <h3 className="font-bold text-sm text-white/60 mb-4 uppercase tracking-wider">Weekly Score History</h3>
            {weekly_scores?.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={weekly_scores}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="label" tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="score" stroke="#3B82F6" strokeWidth={2.5}
                    dot={{ fill: "#3B82F6", r: 4, strokeWidth: 0 }}
                    activeDot={{ r: 6, fill: "#8B5CF6" }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-48 flex items-center justify-center text-white/20 text-sm">
                Complete actions to see score history
              </div>
            )}
          </div>

          {/* Quick nav cards */}
          <div className="space-y-3">
            {navLinks.map(({ href, icon: Icon, label }) => (
              <Link key={href} href={href} className="card-glow p-4 flex items-center gap-3 hover:border-blue-500/30 group">
                <div className="w-9 h-9 rounded-xl bg-blue-500/10 flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                  <Icon size={16} className="text-blue-400" />
                </div>
                <span className="font-medium text-sm">{label}</span>
                <span className="ml-auto text-white/20 group-hover:text-white/50 transition-colors">→</span>
              </Link>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
