"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter, useSearchParams } from "next/navigation";
import { ChevronRight, ChevronLeft, Plus, X, Loader2, Briefcase, Zap, Star, AlertTriangle, Upload } from "lucide-react";
import { onboardUser } from "@/lib/api";
import { useStore } from "@/store/useStore";
import ParticleBackground from "@/components/ParticleBackground";

const steps = ["Profile", "Skills", "Your Role"];

function TagInput({ label, value, onChange, placeholder, icon: Icon, color }: any) {
  const [input, setInput] = useState("");
  const add = () => {
    const v = input.trim();
    if (v && !value.includes(v)) { onChange([...value, v]); setInput(""); }
  };
  return (
    <div>
      <label className="text-sm font-medium text-white/60 mb-2 flex items-center gap-2">
        <Icon size={14} className={color} />{label}
      </label>
      <div className="flex gap-2 mb-2">
        <input value={input} onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), add())}
          placeholder={placeholder} className="input-field" />
        <button type="button" onClick={add} className="btn-ghost px-4 py-3">
          <Plus size={16} />
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {value.map((v: string) => (
          <span key={v} className="tag flex items-center gap-1.5">
            {v}
            <button onClick={() => onChange(value.filter((x: string) => x !== v))} className="hover:text-red-400">
              <X size={11} />
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}

export default function OnboardingPage() {
  const router       = useRouter();
  const params       = useSearchParams();
  const setUser      = useStore((s) => s.setUser);
  const setRole      = useStore((s) => s.setRole);
  const markOnboarded = useStore((s) => s.markOnboarded);

  const userId = params.get("uid") || "";
  const email  = params.get("email") || "";

  const [step,      setStep]      = useState(0);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");
  const [resumeLoading, setResumeLoading] = useState(false);
  const [useResume, setUseResume] = useState(false);  // Toggle between manual and resume
  const [extractedData, setExtractedData] = useState<any>(null); // Store extracted profile data

  // Form state
  const [name,       setName]       = useState("");
  const [phone,      setPhone]      = useState("");
  const [expYears,   setExpYears]   = useState(0);
  const [skills,     setSkills]     = useState<string[]>([]);
  const [strengths,  setStrengths]  = useState<string[]>([]);
  const [weaknesses, setWeaknesses] = useState<string[]>([]);
  const [targetRole, setTargetRole] = useState("");

  // Resume upload handler
  async function handleResumeUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ["application/pdf", "image/png", "image/jpeg"];
    if (!validTypes.includes(file.type)) {
      setError("Invalid file type. Please upload PDF, PNG, or JPG.");
      return;
    }

    setResumeLoading(true);
    setError("");
    try {
      // Create FormData for multipart upload
      const formData = new FormData();
      formData.append("file", file);
      formData.append("user_id", userId || "temp-user");

      console.log("📄 Uploading resume:", file.name);

      // Call API with file upload
      const response = await fetch("http://localhost:8000/api/resume/upload", {
        method: "POST",
        body: formData, // Don't set Content-Type header - browser will set it with boundary
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to extract skills from resume");
      }

      const data = await response.json();
      console.log("✅ Resume analysis result:", data);

      if (data.status === "success") {
        // Extract technical and soft skills from response
        const extractedSkills = data.normalized_skills || [];
        const softSkills = data.soft_skills || [];
        
        console.log(`✓ Extracted ${extractedSkills.length} technical skills from resume`);
        console.log(`✓ Identified ${softSkills.length} soft skills from resume`);

        // Store extracted data for later use
        setExtractedData(data);

        // Auto-populate name and phone if available
        const parsedProfile = data.parsed_profile || {};
        if (parsedProfile.name && !name) {
          setName(parsedProfile.name);
        }
        if (parsedProfile.phone && !phone) {
          setPhone(parsedProfile.phone);
        }
        if (parsedProfile.experience_years && expYears === 0) {
          setExpYears(parsedProfile.experience_years);
        }

        // Add extracted TECHNICAL skills to existing skills (avoid duplicates)
        // Soft skills are kept separate for reference
        const newSkills = Array.from(new Set([...skills, ...extractedSkills]));
        setSkills(newSkills);
        setError(""); // Clear any error
      } else {
        throw new Error(data.message || "Failed to process resume");
      }
    } catch (err: any) {
      console.error("❌ Resume upload error:", err);
      setError(err.message || "Failed to extract skills from resume");
    } finally {
      setResumeLoading(false);
      // Clear file input
      if (e.target) e.target.value = "";
    }
  }

  async function handleSubmit() {
    setLoading(true);
    setError("");
    try {
      // Normalize email to lowercase for consistency
      const normalizedEmail = email.toLowerCase().trim();
      
      const payload = {
        name, email: normalizedEmail, target_role: targetRole,
        skills, strengths, weaknesses,
        experience_years: expYears, phone,
      };
      
      console.log("📤 Onboarding form submitting:", payload);
      console.log("   user_id from URL:", userId);
      console.log("   email from URL:", email);
      console.log("   form target_role:", targetRole);
      
      const result = await onboardUser(payload);
      
      console.log("✅ Onboarding API response:", result);
      console.log("   returned user_id:", result.user_id);
      
      setUser(result.user_id, name);
      setRole(targetRole);
      markOnboarded();
      router.push(`/readiness?uid=${result.user_id}`);
    } catch (err: any) {
      console.error("❌ Onboarding error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const canNext = step === 0
    ? name.trim().length > 0
    : step === 1
    ? skills.length > 0 && strengths.length > 0 && weaknesses.length > 0
    : targetRole.trim().length > 0;

  return (
    <div className="min-h-screen bg-[#0F1117] bg-grid flex items-center justify-center px-4">
      <ParticleBackground />
      <div className="fixed top-20 right-20 w-64 h-64 bg-violet-600/8 rounded-full blur-3xl pointer-events-none" />

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full max-w-lg">
        {/* Progress */}
        <div className="flex items-center gap-2 mb-8 justify-center">
          {steps.map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all
                ${i <= step ? "bg-blue-600 text-white shadow-glow" : "bg-white/[0.06] text-white/30"}`}>
                {i + 1}
              </div>
              <span className={`text-sm ${i === step ? "text-white" : "text-white/30"}`}>{s}</span>
              {i < steps.length - 1 && <div className={`w-8 h-px ${i < step ? "bg-blue-600" : "bg-white/10"}`} />}
            </div>
          ))}
        </div>

        <div className="glass p-8">
          <AnimatePresence mode="wait">
            {step === 0 && (
              <motion.div key="s0" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="space-y-5">
                <h2 className="text-2xl font-black mb-1">Tell us about you</h2>
                <p className="text-white/40 text-sm mb-6">Basic info to personalise your career navigator.</p>
                <div>
                  <label className="block text-sm font-medium text-white/60 mb-2">Full Name *</label>
                  <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Alex Johnson" className="input-field" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-white/60 mb-2">Email</label>
                  <input value={email} disabled className="input-field opacity-50 cursor-not-allowed" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-white/60 mb-2">Phone</label>
                    <input value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="+1 555 000" className="input-field" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-white/60 mb-2">Years of Experience</label>
                    <input type="number" min={0} value={expYears} onChange={(e) => setExpYears(+e.target.value)} className="input-field" />
                  </div>
                </div>
              </motion.div>
            )}

            {step === 1 && (
              <motion.div key="s1" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="space-y-6">
                <h2 className="text-2xl font-black mb-1">Skills & Attributes</h2>
                <p className="text-white/40 text-sm mb-6">Press Enter or click + to add each item.</p>
                
                {/* Resume Upload Toggle */}
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setUseResume(false)}
                    className={`flex-1 px-4 py-2 rounded-lg font-medium transition-all ${
                      !useResume
                        ? "bg-blue-600 text-white shadow-glow"
                        : "bg-white/5 text-white/60 hover:bg-white/10"
                    }`}
                  >
                    Manual Entry
                  </button>
                  <button
                    type="button"
                    onClick={() => setUseResume(true)}
                    className={`flex-1 px-4 py-2 rounded-lg font-medium transition-all ${
                      useResume
                        ? "bg-blue-600 text-white shadow-glow"
                        : "bg-white/5 text-white/60 hover:bg-white/10"
                    }`}
                  >
                    Upload Resume
                  </button>
                </div>

                {/* Resume Upload Section */}
                {useResume && (
                  <div className="space-y-3">
                    <label className="block text-sm font-medium text-white/60">Upload Resume (PDF/TXT)</label>
                    <div className="relative">
                      <input
                        type="file"
                        accept=".pdf,.png,.jpg,.jpeg"
                        onChange={handleResumeUpload}
                        disabled={resumeLoading}
                        className="hidden"
                        id="resume-upload"
                      />
                      <label
                        htmlFor="resume-upload"
                        className={`block w-full p-4 border-2 border-dashed border-blue-500/30 rounded-xl hover:border-blue-500/50 cursor-pointer transition-all text-center ${
                          resumeLoading ? "opacity-50 cursor-not-allowed" : ""
                        }`}
                      >
                        {resumeLoading ? (
                          <div className="flex items-center justify-center gap-2">
                            <Loader2 size={16} className="animate-spin" />
                            Extracting skills...
                          </div>
                        ) : (
                          <div className="text-white/60">
                            <Upload size={20} className="mx-auto mb-2 opacity-60" />
                            Click to upload resume (PDF/PNG/JPG)
                          </div>
                        )}
                      </label>
                    </div>
                    {skills.length > 0 && (
                      <div className="text-sm text-green-400 flex items-center gap-2">
                        ✓ {skills.length} skills extracted
                      </div>
                    )}
                    
                    {extractedData && (
                      <div className="mt-4 space-y-3">
                        {/* Profile Data */}
                        <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-sm text-white/70 space-y-1">
                          <div className="font-semibold text-blue-400 mb-2">📋 Extracted Profile Data</div>
                          {extractedData.parsed_profile?.name && (
                            <div>Name: <span className="text-white">{extractedData.parsed_profile.name}</span></div>
                          )}
                          {extractedData.parsed_profile?.email && (
                            <div>Email: <span className="text-white">{extractedData.parsed_profile.email}</span></div>
                          )}
                          {extractedData.parsed_profile?.experience_years && (
                            <div>Experience: <span className="text-white">{extractedData.parsed_profile.experience_years} years</span></div>
                          )}
                          {extractedData.parsed_profile?.education && extractedData.parsed_profile.education.length > 0 && (
                            <div>Education: <span className="text-white">
                              {typeof extractedData.parsed_profile.education[0] === 'string'
                                ? extractedData.parsed_profile.education[0]
                                : `${extractedData.parsed_profile.education[0].degree || ''} from ${extractedData.parsed_profile.education[0].institution || ''}`
                              }
                            </span></div>
                          )}
                        </div>
                        
                        {/* Technical Skills Summary */}
                        {extractedData.normalized_skills && extractedData.normalized_skills.length > 0 && (
                          <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-sm text-white/70">
                            <div className="font-semibold text-green-400 mb-2">⚙️ Technical Skills ({extractedData.normalized_skills.length})</div>
                            <div className="flex flex-wrap gap-1.5">
                              {extractedData.normalized_skills.slice(0, 8).map((skill: string) => (
                                <span key={skill} className="px-2 py-0.5 bg-green-500/20 text-green-300 rounded text-xs">
                                  {skill.charAt(0).toUpperCase() + skill.slice(1)}
                                </span>
                              ))}
                              {extractedData.normalized_skills.length > 8 && (
                                <span className="px-2 py-0.5 bg-green-500/10 text-green-300 rounded text-xs">
                                  +{extractedData.normalized_skills.length - 8} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {/* Soft Skills Summary */}
                        {extractedData.soft_skills && extractedData.soft_skills.length > 0 && (
                          <div className="p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg text-sm text-white/70">
                            <div className="font-semibold text-purple-400 mb-2">💡 Soft Skills ({extractedData.soft_skills.length})</div>
                            <div className="flex flex-wrap gap-1.5">
                              {extractedData.soft_skills.slice(0, 6).map((skill: string) => (
                                <span key={skill} className="px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded text-xs">
                                  {skill.charAt(0).toUpperCase() + skill.slice(1)}
                                </span>
                              ))}
                              {extractedData.soft_skills.length > 6 && (
                                <span className="px-2 py-0.5 bg-purple-500/10 text-purple-300 rounded text-xs">
                                  +{extractedData.soft_skills.length - 6} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Current Skills Display */}
                {skills.length > 0 && (
                  <div>
                    <label className="text-sm font-medium text-white/60 mb-2 flex items-center gap-2">
                      <Zap size={14} className="text-blue-400" />Current Skills
                    </label>
                    <div className="flex flex-wrap gap-2 mb-3">
                      {skills.map((v: string) => (
                        <span key={v} className="tag flex items-center gap-1.5">
                          {v}
                          <button onClick={() => setSkills(skills.filter((x: string) => x !== v))} className="hover:text-red-400">
                            <X size={11} />
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Manual Skills Input */}
                {!useResume && (
                  <TagInput label="Current Skills" value={skills} onChange={setSkills} placeholder="e.g. Python, React" icon={Zap} color="text-blue-400" />
                )}

                <TagInput label="Strengths" value={strengths} onChange={setStrengths} placeholder="e.g. problem-solving" icon={Star} color="text-yellow-400" />
                <TagInput label="Weaknesses" value={weaknesses} onChange={setWeaknesses} placeholder="e.g. SQL, public speaking" icon={AlertTriangle} color="text-orange-400" />
              </motion.div>
            )}

            {step === 2 && (
              <motion.div key="s2" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="space-y-5">
                <h2 className="text-2xl font-black mb-1">Your Target Role</h2>
                <p className="text-white/40 text-sm mb-6">What career position are you aiming for?</p>
                <div>
                  <label className="text-sm font-medium text-white/60 mb-2 flex items-center gap-2">
                    <Briefcase size={14} className="text-violet-400" /> Target Role *
                  </label>
                  <input value={targetRole} onChange={(e) => setTargetRole(e.target.value)}
                    placeholder="e.g. Full Stack Developer, Data Scientist" className="input-field" />
                </div>
                {error && (
                  <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-xl p-3">{error}</div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Nav buttons */}
          <div className="flex justify-between mt-8">
            <button onClick={() => setStep(s => s - 1)} disabled={step === 0} className="btn-ghost flex items-center gap-2 disabled:opacity-0">
              <ChevronLeft size={16} /> Back
            </button>
            {step < steps.length - 1 ? (
              <button onClick={() => setStep(s => s + 1)} disabled={!canNext} className="btn-primary flex items-center gap-2">
                Next <ChevronRight size={16} />
              </button>
            ) : (
              <button onClick={handleSubmit} disabled={!canNext || loading} className="btn-primary flex items-center gap-2">
                {loading ? <><Loader2 size={16} className="animate-spin" /> Creating...</> : <>Launch Navigator <ChevronRight size={16} /></>}
              </button>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
