"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Brain, ChevronDown, ChevronUp,
  Mail, Target, ClipboardList, BarChart3, Map,
  MessageSquare, RefreshCw, TrendingUp, Upload,
  HelpCircle, BookOpen,
  CheckCircle, Star, Users,
} from "lucide-react";
import Image from "next/image";
import { useStore } from "@/store/useStore";
import ParticleBackground from "@/components/ParticleBackground";

// ── Types ─────────────────────────────────────────────────────────
interface FAQItem { q: string; a: string; }
interface Section {
  id: string;
  icon: any;
  color: string;
  glow: string;
  title: string;
  subtitle: string;
  steps?: { icon?: any; title: string; desc: string }[];
  faqs?: FAQItem[];
  tips?: string[];
}

// ── FAQ accordion ─────────────────────────────────────────────────
function FAQ({ items }: { items: FAQItem[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="glass rounded-xl overflow-hidden border border-white/[0.06]">
          <button
            onClick={() => setOpen(open === i ? null : i)}
            className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-white/[0.02] transition-colors"
          >
            <span className="font-medium text-sm text-white/80 pr-4">{item.q}</span>
            {open === i
              ? <ChevronUp size={15} className="text-blue-400 shrink-0" />
              : <ChevronDown size={15} className="text-white/30 shrink-0" />}
          </button>
          <AnimatePresence initial={false}>
            {open === i && (
              <motion.div
                key="ans"
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <p className="px-5 pb-4 text-sm text-white/50 leading-relaxed border-t border-white/[0.05] pt-3">
                  {item.a}
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ))}
    </div>
  );
}

// ── Step card ─────────────────────────────────────────────────────
function StepCard({ index, title, desc, icon: Icon }: { index: number; title: string; desc: string; icon?: any }) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-sm font-black text-white shrink-0 shadow-glow">
          {index}
        </div>
        <div className="w-px flex-1 bg-white/[0.06] mt-2" />
      </div>
      <div className="pb-6">
        <div className="flex items-center gap-2 mb-1">
          {Icon && <Icon size={14} className="text-blue-400" />}
          <h4 className="font-semibold text-white text-sm">{title}</h4>
        </div>
        <p className="text-white/40 text-sm leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}

// ── Tip pill ──────────────────────────────────────────────────────
function TipPill({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2.5 p-3 rounded-xl bg-blue-500/5 border border-blue-500/10">
      <Star size={13} className="text-blue-400 mt-0.5 shrink-0" />
      <p className="text-sm text-white/60 leading-relaxed">{text}</p>
    </div>
  );
}

// ── Data ──────────────────────────────────────────────────────────
const sections: Section[] = [
  {
    id: "getting-started",
    icon: BookOpen,
    color: "text-blue-400",
    glow: "bg-blue-500/15",
    title: "Getting Started",
    subtitle: "From sign-up to your first roadmap in minutes.",
    steps: [
      { icon: Mail,          title: "Enter your email",         desc: "Open the app and enter your email address on the sign-in page. If you are new, an account is created and a password is sent to your email. If you already have an account, just enter your password." },
      { icon: Users,         title: "Complete onboarding",      desc: "Fill in your name, years of experience, skills, strengths, and weaknesses across 3 short steps. You can also upload a PDF or image resume to auto-fill your skills." },
      { icon: Target,        title: "Set your target role",     desc: "Enter the specific job title you are aiming for — e.g. 'Full Stack Developer', 'Data Scientist', 'DevOps Engineer'. Be specific for best results." },
      { icon: ClipboardList, title: "Take the readiness assessment", desc: "Answer 10 AI-generated questions about your target role. Take your time — detailed answers produce a more accurate score. You can go back and edit any answer before submitting." },
      { icon: CheckCircle,   title: "Review your result",       desc: "See your readiness score (0–100), your qualification status, and a list of safer and advanced adjacent roles. Then head to your Dashboard." },
    ],
  },
  {
    id: "dashboard",
    icon: BarChart3,
    color: "text-violet-400",
    glow: "bg-violet-500/15",
    title: "Dashboard",
    subtitle: "Your command centre — everything at a glance.",
    steps: [
      { title: "Skill Level",       desc: "Derived from your confidence score. LOW (0–29) means you need more practice, MEDIUM (30–79) means steady progress, HIGH (80–100) means you are ready to apply." },
      { title: "Confidence Score",  desc: "Starts at your readiness score and updates ±1 point after every action assessment you complete. It never lies — it is a live reflection of your demonstrated ability." },
      { title: "Progress",          desc: "Percentage of roadmap actions completed. Each action you pass or fail counts. Come back regularly and work through the roadmap month by month." },
      { title: "Score History",     desc: "The line chart shows your score for every action you have assessed so far. A rising trend means you are building real skill. A plateau means it is time to revisit weak areas." },
    ],
    faqs: [
      { q: "Why is my confidence score low?", a: "Your confidence starts at your readiness score. If you scored low on assessment or failed several actions, the score reflects that. Focus on completing roadmap actions and re-assessing to build it back up." },
      { q: "Can I retake the readiness assessment?", a: "Yes — use the Readiness link from your dashboard at any time. Each retake generates a fresh set of AI questions. Your previous score will be overwritten." },
    ],
  },
  {
    id: "roadmap",
    icon: Map,
    color: "text-emerald-400",
    glow: "bg-emerald-500/15",
    title: "5-Month Roadmap",
    subtitle: "Your personalised action plan, month by month.",
    steps: [
      { title: "Structure",           desc: "5 months × 4 actions = 20 total actions. Each action is a specific skill or task relevant to your target role, generated by the RoadmapAgent based on your profile." },
      { title: "Start an assessment", desc: "Click 'Start Assessment' on any pending action. The AI generates 10 questions specific to that action and your role. Answer all 10 and submit." },
      { title: "Scoring",             desc: "Score ≥ 50 = Passed (confidence +1). Score < 50 = Failed (confidence -1). A failed action can be reattempted — just click 'Start Assessment' again." },
      { title: "Work in order",       desc: "Month 1 builds foundations for Month 2 and so on. It is best to complete each month before moving to the next, but the system does not enforce this." },
    ],
    tips: [
      "Write detailed answers — the AI scores depth and accuracy, not just keywords.",
      "Failed actions are not permanent. Revisit them after studying the topic and try again.",
      "All 5 months are visible from day one — you can peek ahead to prepare.",
    ],
    faqs: [
      { q: "What if my roadmap does not match my actual skills?", a: "Use the Regenerate option on the roadmap page (or reroute to a new role). You can also update your profile via onboarding again to give the AI more context." },
      { q: "What happens to my progress if I regenerate the roadmap?", a: "Action scores are reset to pending. Your confidence score is kept. The new roadmap is tailored to your updated profile." },
    ],
  },
  {
    id: "market",
    icon: BarChart3,
    color: "text-yellow-400",
    glow: "bg-yellow-500/15",
    title: "Market Intelligence",
    subtitle: "Live demand data for your target role.",
    steps: [
      { title: "Demand Score",         desc: "A score out of 100 representing how in-demand your target role is in the current market. Above 70 is strong demand." },
      { title: "Salary Range",         desc: "Average salary range in USD for your role. This is an AI estimate based on current market data — use it as a reference, not a guarantee." },
      { title: "In-Demand Skills",     desc: "The specific technical skills employers are actively hiring for in your role right now. Cross-reference these with your roadmap actions." },
      { title: "Adjacent Safer Roles", desc: "Roles with similar skill requirements but lower competition. If your confidence is low, these are stepping-stone options worth considering." },
      { title: "Key Hiring Companies", desc: "A snapshot of companies actively hiring for your role. Use this for targeted job applications." },
    ],
    tips: [
      "Market data is regenerated each time you onboard or switch roles — visit this page after updating your target role.",
      "Cross-check In-Demand Skills with your current skills to identify your biggest gaps.",
    ],
  },
  {
    id: "hands-on",
    icon: MessageSquare,
    color: "text-cyan-400",
    glow: "bg-cyan-500/15",
    title: "Hands-On Mentor",
    subtitle: "Learn by doing with your AI technical mentor.",
    steps: [
      { title: "Start a session",   desc: "Click 'Start Session'. The mentor will assign you one real-world practical task tailored to your target role. No setup needed." },
      { title: "Follow the steps",  desc: "The mentor breaks the task into clear steps. Complete each one in your actual development environment, then report back." },
      { title: "Ask questions",     desc: "Stuck? Type your question and the mentor will clarify, provide examples, or suggest resources. It has full memory of the conversation." },
      { title: "End the session",   desc: "Type $ to end the session when you are done. Each new session starts fresh with a new task." },
    ],
    tips: [
      "Actually do the tasks in a real editor or terminal — the mentor is designed for hands-on learning, not theory.",
      "The more context you give in your replies (e.g. error messages, code snippets), the more precise the guidance.",
      "Type 'explain' after any step if you want a deeper explanation before proceeding.",
    ],
    faqs: [
      { q: "Does the hands-on mentor remember previous sessions?", a: "No — each session starts fresh. The mentor has memory within a single session only. This keeps tasks focused and avoids context overload." },
    ],
  },
  {
    id: "reroute",
    icon: RefreshCw,
    color: "text-orange-400",
    glow: "bg-orange-500/15",
    title: "Career Rerouting",
    subtitle: "Smart path adjustment when confidence drops.",
    steps: [
      { title: "When it triggers",         desc: "The rerouting check runs automatically after every action assessment. You will see an alert on the dashboard when confidence drops below 40." },
      { title: "Reroute Suggested",        desc: "If the AI suggests a reroute, you will see a list of safer adjacent roles. These have lower competition and similar skills — switching gives you a shorter path to employment." },
      { title: "Switch roles",             desc: "Clicking 'Switch' on a suggested role will update your profile, regenerate your roadmap for the new role, and refresh your market analysis." },
      { title: "Return to previous role",  desc: "Once confidence reaches ≥ 80 on the new role, a 'Return to Previous Role' option appears. You can switch back and pick up where you left off." },
      { title: "On Track",                 desc: "If the AI says you are on track, no action is needed. Keep working through your roadmap." },
    ],
    tips: [
      "Rerouting is not failure — it is a smart recalibration. Many successful careers take a lateral step before going up.",
      "You can manually open the Reroute page from your dashboard at any time to check your status, even without a low confidence score.",
    ],
    faqs: [
      { q: "Does switching roles delete my previous roadmap?", a: "No. Your original roadmap and progress data are preserved in the database. When you return to your previous role, the original roadmap is restored." },
      { q: "Can I ignore a reroute suggestion?", a: "Yes. Rerouting is always optional. You can dismiss the alert and continue working on your current roadmap." },
    ],
  },
  {
    id: "feedback",
    icon: TrendingUp,
    color: "text-pink-400",
    glow: "bg-pink-500/15",
    title: "Feedback Report",
    subtitle: "A full AI analysis of your progress.",
    steps: [
      { title: "Overall Rating",       desc: "A qualitative assessment of your overall progress: Excellent, Good, Moderate, or Needs Improvement — based on actions completed, scores, and confidence trend." },
      { title: "Velocity",             desc: "How fast you are moving through the roadmap compared to the expected pace. Slow velocity means you may need to increase the frequency of your sessions." },
      { title: "Strengths Observed",   desc: "Specific areas where your assessment scores show consistent competence. These are your current strongest selling points." },
      { title: "Areas of Concern",     desc: "Topics where your scores are low or inconsistent. These need dedicated study before your next assessment attempt." },
      { title: "Learning Insights",    desc: "Each insight includes evidence from your data, a specific recommendation, and a priority level. Act on high-priority insights first." },
      { title: "Recommended Adjustments", desc: "Concrete changes to your plan — e.g. spending more time on a specific skill, adjusting pace, or switching study resources." },
    ],
    tips: [
      "Regenerate feedback regularly — it reflects your current state and improves as you complete more actions.",
      "Focus on the Areas of Concern before attempting more assessments to avoid repeated failures.",
    ],
  },
  {
    id: "resume",
    icon: Upload,
    color: "text-indigo-400",
    glow: "bg-indigo-500/15",
    title: "Resume Upload",
    subtitle: "Auto-fill your profile from your CV.",
    steps: [
      { title: "Supported formats",   desc: "PDF (recommended), PNG, JPG. Maximum recommended file size is 5 MB. Scanned images are supported via OCR but plain-text PDFs give the most accurate extraction." },
      { title: "What gets extracted", desc: "Full name, phone number, years of experience, technical skills (Python, React, SQL, etc.), and soft skills (communication, leadership, etc.)." },
      { title: "Review before saving", desc: "Extracted data auto-fills the onboarding form. Review every field before submitting — AI extraction is highly accurate but not perfect, especially for non-standard CV layouts." },
      { title: "Add manually",         desc: "Any skills the AI missed can be added manually via the tag input. The two approaches work together — you can upload a resume and then add extra skills by hand." },
    ],
    tips: [
      "Use a clean, text-based PDF for best extraction accuracy. Heavily styled or image-only PDFs may miss some data.",
      "If your resume extraction fails, switch to Manual Entry and fill in your skills directly — it takes about 2 minutes.",
    ],
    faqs: [
      { q: "Is my resume stored permanently?", a: "Uploaded resume files are saved securely. The extracted data (skills, profile) is saved to your account. The raw file is kept for reference only." },
    ],
  },
];

// ── Sidebar nav item ──────────────────────────────────────────────
function SideItem({ s, active, onClick }: { s: Section; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all
        ${active
          ? "bg-blue-500/10 border border-blue-500/20 text-white"
          : "text-white/40 hover:text-white/70 hover:bg-white/[0.04]"}`}
    >
      <div className={`w-8 h-8 rounded-lg ${active ? s.glow : "bg-white/[0.04]"} flex items-center justify-center shrink-0`}>
        <s.icon size={15} className={active ? s.color : "text-white/30"} />
      </div>
      <span className="text-sm font-medium">{s.title}</span>
    </button>
  );
}

// ── Main page ─────────────────────────────────────────────────────
export default function HelpPage() {
  const [active, setActive] = useState("getting-started");
  const section = sections.find((s) => s.id === active)!;
  const userId  = useStore((s) => s.userId);

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid">
      <ParticleBackground />

      {/* Top nav */}
      <nav className="glass border-b border-white/[0.06] sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
              <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center">
                <Image src="/nexus_logo.png" alt="Nexus-AI" width={32} height={32} className="object-contain" />
              </div>
              <span className="font-bold">Nexus-AI</span>
            </Link>
            <span className="text-white/20">/</span>
            <span className="text-white/50 text-sm font-medium flex items-center gap-1.5">
              <HelpCircle size={14} /> Help Center
            </span>
          </div>
          <button onClick={() => window.history.back()} className="btn-ghost text-sm py-2 px-4">
            ← Back
          </button>
        </div>
      </nav>

      {/* Hero banner */}
      <div className="max-w-7xl mx-auto px-6 pt-10 pb-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="glass p-8 rounded-2xl border border-white/[0.08] flex flex-col md:flex-row items-start md:items-center gap-6"
        >
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-glow shrink-0">
            <HelpCircle size={26} className="text-white" />
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-black mb-1">Help Center</h1>
            <p className="text-white/40 text-sm leading-relaxed">
              Everything you need to get the most out of Nexus-AI — from your first login to advanced rerouting strategies.
              Pick a topic from the sidebar to get started.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 shrink-0">
            {[
              { label: "7 Topics", icon: BookOpen },
              { label: "Guides + FAQs", icon: CheckCircle },
              { label: "Tips Included", icon: Star },
            ].map(({ label, icon: Icon }) => (
              <div key={label} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/[0.06] border border-white/[0.08] text-xs text-white/50">
                <Icon size={11} className="text-blue-400" /> {label}
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Body */}
      <div className="max-w-7xl mx-auto px-6 pb-16 flex gap-6">

        {/* Sidebar */}
        <aside className="w-56 shrink-0 hidden md:block">
          <div className="sticky top-24 space-y-1">
            {sections.map((s) => (
              <SideItem key={s.id} s={s} active={active === s.id} onClick={() => setActive(s.id)} />
            ))}
          </div>
        </aside>

        {/* Mobile section picker */}
        <div className="md:hidden w-full mb-4">
          <select
            value={active}
            onChange={(e) => setActive(e.target.value)}
            className="input-field w-full"
          >
            {sections.map((s) => (
              <option key={s.id} value={s.id}>{s.title}</option>
            ))}
          </select>
        </div>

        {/* Main content */}
        <main className="flex-1 min-w-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={active}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="space-y-6"
            >
              {/* Section header */}
              <div className="flex items-center gap-4">
                <div className={`w-12 h-12 rounded-xl ${section.glow} flex items-center justify-center`}>
                  <section.icon size={22} className={section.color} />
                </div>
                <div>
                  <h2 className="text-2xl font-black">{section.title}</h2>
                  <p className="text-white/40 text-sm mt-0.5">{section.subtitle}</p>
                </div>
              </div>

              {/* Steps */}
              {section.steps && section.steps.length > 0 && (
                <div className="card-glow p-6">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-white/40 mb-5">
                    {section.id === "getting-started" ? "How it works" : "Understanding the features"}
                  </h3>
                  <div>
                    {section.steps.map((step, i) => (
                      <StepCard key={i} index={i + 1} title={step.title} desc={step.desc} icon={step.icon} />
                    ))}
                  </div>
                </div>
              )}

              {/* Tips */}
              {section.tips && section.tips.length > 0 && (
                <div className="card-glow p-6">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-white/40 mb-4 flex items-center gap-2">
                    <Star size={12} className="text-yellow-400" /> Pro Tips
                  </h3>
                  <div className="space-y-2">
                    {section.tips.map((t, i) => <TipPill key={i} text={t} />)}
                  </div>
                </div>
              )}

              {/* FAQs */}
              {section.faqs && section.faqs.length > 0 && (
                <div className="card-glow p-6">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-white/40 mb-4">
                    Frequently Asked Questions
                  </h3>
                  <FAQ items={section.faqs} />
                </div>
              )}

              {/* Bottom nav */}
              <div className="flex items-center justify-between pt-2">
                {sections.findIndex((s) => s.id === active) > 0 ? (
                  <button
                    onClick={() => setActive(sections[sections.findIndex((s) => s.id === active) - 1].id)}
                    className="btn-ghost flex items-center gap-2 text-sm"
                  >
                    ← Previous
                  </button>
                ) : <div />}
                {sections.findIndex((s) => s.id === active) < sections.length - 1 ? (
                  <button
                    onClick={() => setActive(sections[sections.findIndex((s) => s.id === active) + 1].id)}
                    className="btn-primary flex items-center gap-2 text-sm"
                  >
                    Next Topic →
                  </button>
                ) : (
                  <Link href={userId ? "/dashboard" : "/auth"} className="btn-primary flex items-center gap-2 text-sm">
                    {userId ? "Go to Dashboard →" : "Get Started →"}
                  </Link>
                )}
              </div>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
