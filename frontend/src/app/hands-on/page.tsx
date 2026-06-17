"use client";
import { useState, useRef, useEffect } from "react";
import { Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSearchParams } from "next/navigation";
import { Send, Bot, User, Loader2, ArrowLeft, Zap } from "lucide-react";
import { handsOnChat } from "@/lib/api";
import { useStore } from "@/store/useStore";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import ParticleBackground from "@/components/ParticleBackground";

interface Message { role: "user" | "assistant"; content: string; }

const sectionLabels = ["Task", "Current Step", "What To Do", "Reply After Completion"];

function formatInline(text: string) {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index} className="rounded bg-black/30 px-1.5 py-0.5 font-mono text-[0.85em] text-blue-200">{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function AssistantMessage({ content }: { content: string }) {
  const normalized = sectionLabels.reduce(
    (text, label) => text.replace(new RegExp(`^[ \\t]*(?:\\*\\*)?${label}:(?:\\*\\*)?[ \\t]*`, "gm"), `${label}:\n`),
    content,
  ).trim();
  const lines = normalized.split("\n");
  let inCodeBlock = false;

  return (
    <div className="space-y-2.5">
      {lines.map((rawLine, index) => {
        const line = rawLine.trim();

        if (line.startsWith("```")) {
          inCodeBlock = !inCodeBlock;
          return null;
        }
        if (inCodeBlock) {
          return <pre key={index} className="overflow-x-auto rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-xs text-blue-100"><code>{rawLine}</code></pre>;
        }
        if (!line) return null;

        const section = sectionLabels.find((label) => line.toLowerCase() === `${label.toLowerCase()}:`);
        if (section) {
          return (
            <div key={index} className="flex items-center gap-2 pt-2 first:pt-0">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
              <h3 className="text-xs font-bold uppercase tracking-[0.14em] text-blue-300">{section}</h3>
            </div>
          );
        }

        const heading = line.match(/^#{1,3}\s+(.+)$/);
        if (heading) return <h3 key={index} className="pt-1 font-semibold text-white">{formatInline(heading[1])}</h3>;

        const listItem = line.match(/^[-*\u2022]\s+(.+)$/);
        if (listItem) {
          return <div key={index} className="flex gap-2 pl-1"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-white/50" /><p>{formatInline(listItem[1])}</p></div>;
        }

        const numberedItem = line.match(/^(\d+)[.)]\s+(.+)$/);
        if (numberedItem) {
          return <div key={index} className="flex gap-2"><span className="flex h-5 min-w-5 items-center justify-center rounded-md bg-blue-500/15 px-1 text-[10px] font-bold text-blue-300">{numberedItem[1]}</span><p>{formatInline(numberedItem[2])}</p></div>;
        }

        return <p key={index} className="whitespace-pre-wrap">{formatInline(line)}</p>;
      })}
    </div>
  );
}

export default function HandsOnPage() {
  return <Suspense fallback={<div className="min-h-screen bg-[#0F1117] flex items-center justify-center"><Loader2 size={28} className="animate-spin text-blue-400" /></div>}><HandsOnPageInner /></Suspense>;
}

function HandsOnPageInner() {
  const userId = useAuthGuard();

  const [messages, setMessages] = useState<Message[]>([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const [started,  setStarted]  = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function startSession() {
    setStarted(true);
    setLoading(true);
    try {
      const res = await handsOnChat({
        user_id: userId,
        message: "Assign my first hands-on practical task.",
        conversation_history: [],
      });
      setMessages(res.conversation_history as Message[]);
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  async function sendMessage() {
    if (!input.trim() || loading) return;
    const msg = input.trim();
    setInput("");
    setLoading(true);

    // Optimistically add user message
    const history = [...messages];
    setMessages(prev => [...prev, { role: "user", content: msg }]);

    try {
      const res = await handsOnChat({
        user_id: userId,
        message: msg,
        conversation_history: history,
      });
      setMessages(res.conversation_history as Message[]);
    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { role: "assistant", content: "Something went wrong. Please try again." }]);
    }
    setLoading(false);
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid flex flex-col">
      <ParticleBackground />
      {/* Header */}
      <div className="glass border-b border-white/[0.06] px-6 py-4 flex items-center gap-3">
        <button onClick={() => window.history.back()} className="btn-ghost p-2"><ArrowLeft size={16} /></button>
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-glow">
          <Zap size={16} className="text-white" />
        </div>
        <div>
          <h1 className="font-bold">Hands-On AI Mentor</h1>
          <p className="text-white/30 text-xs">Practical task-based learning · Type $ to end session</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 max-w-3xl mx-auto w-full space-y-4">
        {!started && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="glass p-8 text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center mx-auto mb-4 shadow-glow">
              <Bot size={28} className="text-white" />
            </div>
            <h2 className="text-2xl font-black mb-2">Hands-On Mentor</h2>
            <p className="text-white/40 leading-relaxed mb-6">
              Your AI mentor will assign you a real-world task tailored to your target role.
              Follow the steps, ask questions, and complete the task hands-on.
            </p>
            <button onClick={startSession} disabled={loading}
              className="btn-primary inline-flex items-center gap-2 px-8 py-3">
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
              Start Session
            </button>
          </motion.div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5
                ${msg.role === "user"
                  ? "bg-blue-600"
                  : "bg-gradient-to-br from-violet-500 to-blue-600"}`}>
                {msg.role === "user" ? <User size={14} /> : <Bot size={14} />}
              </div>
              <div className={`max-w-[78%] px-4 py-3 rounded-2xl text-sm leading-relaxed
                ${msg.role === "user"
                  ? "bg-blue-600 text-white rounded-tr-sm"
                  : "glass rounded-tl-sm text-white/80"}`}>
                {msg.role === "assistant"
                  ? <AssistantMessage content={msg.content} />
                  : <p className="whitespace-pre-wrap">{msg.content}</p>}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && started && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-blue-600 flex items-center justify-center">
              <Bot size={14} />
            </div>
            <div className="glass px-4 py-3 rounded-2xl rounded-tl-sm">
              <div className="flex gap-1 items-center h-5">
                {[0, 1, 2].map(i => (
                  <motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-white/40"
                    animate={{ y: [-3, 0, -3] }}
                    transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }} />
                ))}
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      {started && (
        <div className="glass border-t border-white/[0.06] px-4 py-4">
          <div className="max-w-3xl mx-auto flex gap-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder='Type your response... (Enter to send, Shift+Enter for newline, $ to end)'
              rows={2}
              className="input-field flex-1 resize-none"
            />
            <button onClick={sendMessage} disabled={!input.trim() || loading}
              className="btn-primary px-4 self-end">
              <Send size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
