"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import { Brain, Github, Linkedin, Mail, Star, Code2, Heart, ArrowLeft } from "lucide-react";
import Image from "next/image";
import { useStore } from "@/store/useStore";
import ParticleBackground from "@/components/ParticleBackground";

const team = [
  {
    name: "Deepak Thillai",
    role: "Head Developer",
    lead: true,
    bio: "Architected and built the full-stack Nexus-AI platform — FastAPI backend, Next.js frontend, AI agent pipeline, and MongoDB integration.",
    links: {
      linkedin: "https://www.linkedin.com/in/deepak-thillai-6901a3305/",
      github: "https://github.com/DeepakThillai",
      email: "deepakthillaikannu@gmail.com",
    },
  },
  {
    name: "Abdullah N",
    role: "Developer",
    lead: false,
    bio: "Contributed to the development and implementation of core features across the Nexus-AI platform.",
    links: {
      github: "https://github.com/Abdullah-218",
    },
  },
  {
    name: "Baranidharan P",
    role: "Developer",
    lead: false,
    bio: "Contributed to the development and implementation of core features across the Nexus-AI platform.",
    links: {
      github: "https://github.com/barani961",
    },
  },
];

const stack = [
  { label: "Frontend",  value: "Next.js 14, TypeScript, Tailwind CSS, Framer Motion" },
  { label: "Backend",   value: "FastAPI, Python, Uvicorn" },
  { label: "AI",        value: "Groq API — RAG" },
  { label: "Database",  value: "MongoDB Atlas" },
  { label: "State",     value: "Zustand, Recharts" },
];

export default function CreditsPage() {
  const userId = useStore((s) => s.userId);
  const appLink  = userId ? `/dashboard` : "/";
  const appLabel = userId ? "← Dashboard" : "← Back";
  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid">
      <ParticleBackground />
      <div className="fixed top-1/4 left-1/4 w-96 h-96 bg-blue-600/6 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed bottom-1/4 right-1/4 w-80 h-80 bg-violet-600/6 rounded-full blur-3xl pointer-events-none" />

      {/* Nav */}
      <nav className="glass border-b border-white/[0.06] sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
            <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center">
              <Image src="/nexus_logo.png" alt="Nexus-AI" width={32} height={32} className="object-contain" />
            </div>
            <span className="font-bold">Nexus-AI</span>
          </Link>
          <div className="flex items-center gap-2">
            <Link href="/help" className="btn-ghost text-sm py-2 px-4">Help</Link>
            <Link href={appLink} className="btn-ghost text-sm py-2 px-4">{appLabel}</Link>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-6 py-16 space-y-16">

        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-300 text-sm font-medium mb-6">
            <Heart size={13} className="text-violet-400" /> Built with passion
          </div>
          <h1 className="text-5xl md:text-6xl font-black tracking-tight mb-4">
            Meet the <span className="gradient-text">Team</span>
          </h1>
          <p className="text-white/40 text-lg max-w-xl mx-auto leading-relaxed">
            Nexus-AI was designed and built by a small team passionate about making career navigation intelligent and accessible.
          </p>
        </motion.div>

        {/* Team cards */}
        <div className="space-y-5">
          {team.map((member, i) => (
            <motion.div
              key={member.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              className={`glass p-7 rounded-2xl border flex flex-col md:flex-row md:items-center gap-6
                ${member.lead
                  ? "border-blue-500/25 bg-gradient-to-br from-blue-500/[0.06] to-violet-500/[0.06]"
                  : "border-white/[0.08]"}`}
            >
              {/* Avatar */}
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-black shrink-0
                ${member.lead
                  ? "bg-gradient-to-br from-blue-500 to-violet-600 shadow-glow"
                  : "bg-white/[0.06] border border-white/[0.1]"}`}
              >
                <span className={member.lead ? "text-white" : "text-white/60"}>
                  {member.name.charAt(0)}
                </span>
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-2.5 mb-1">
                  <h2 className="text-xl font-black">{member.name}</h2>
                  {member.lead && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-500/15 text-blue-300 border border-blue-500/20">
                      <Star size={10} /> Head Developer
                    </span>
                  )}
                  {!member.lead && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-white/[0.06] text-white/40 border border-white/[0.08]">
                      <Code2 size={10} /> Developer
                    </span>
                  )}
                </div>
                <p className="text-white/40 text-sm leading-relaxed">{member.bio}</p>
              </div>

              {/* Links */}
              <div className="flex items-center gap-2 shrink-0">
                {member.links.linkedin && (
                  <a
                    href={member.links.linkedin}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center hover:bg-blue-500/20 transition-colors group"
                    title="LinkedIn"
                  >
                    <Linkedin size={16} className="text-blue-400 group-hover:text-blue-300" />
                  </a>
                )}
                {member.links.github && (
                  <a
                    href={member.links.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 rounded-xl bg-white/[0.06] border border-white/[0.08] flex items-center justify-center hover:bg-white/[0.1] transition-colors group"
                    title="GitHub"
                  >
                    <Github size={16} className="text-white/50 group-hover:text-white/80" />
                  </a>
                )}
                {member.links.email && (
                  <a
                    href={`mailto:${member.links.email}`}
                    className="w-10 h-10 rounded-xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center hover:bg-violet-500/20 transition-colors group"
                    title={member.links.email}
                  >
                    <Mail size={16} className="text-violet-400 group-hover:text-violet-300" />
                  </a>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Built with */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="card-glow p-7 rounded-2xl"
        >
          <h3 className="text-lg font-black mb-5 flex items-center gap-2">
            <Code2 size={18} className="text-blue-400" /> Built With
          </h3>
          <div className="space-y-3">
            {stack.map(({ label, value }) => (
              <div key={label} className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-4 py-2.5 border-b border-white/[0.05] last:border-0">
                <span className="text-xs font-bold uppercase tracking-wider text-white/30 w-24 shrink-0">{label}</span>
                <span className="text-sm text-white/60">{value}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Contact */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="glass p-8 rounded-2xl border border-blue-500/20 bg-gradient-to-br from-blue-500/[0.05] to-violet-500/[0.05] text-center"
        >
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center mx-auto mb-4 shadow-glow">
            <Mail size={20} className="text-white" />
          </div>
          <h3 className="text-xl font-black mb-2">Contact Us</h3>
          <p className="text-white/40 text-sm mb-6 max-w-sm mx-auto">
            Have a question, suggestion, or want to collaborate? We'd love to hear from you.
          </p>
          <a
            href="mailto:deepakthillaikannu@gmail.com"
            className="btn-primary inline-flex items-center gap-2 px-8 py-3"
          >
            <Mail size={15} /> deepakthillaikannu@gmail.com
          </a>
        </motion.div>

        {/* Footer note */}
        <p className="text-center text-white/20 text-sm pb-4">
          © 2026 Nexus-AI — Made with <Heart size={12} className="inline text-red-400/60" /> by the Nexus-AI team
        </p>

      </main>
    </div>
  );
}
