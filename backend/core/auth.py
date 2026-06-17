"""
auth.py — Password generation, hashing, verification, and email sending.
Uses Brevo HTTP API (port 443) — SMTP is blocked on Render free tier.
"""
import os
import random
import string
import json
import urllib.request
import urllib.error
from pathlib import Path

import bcrypt

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# ── Password helpers ──────────────────────────────────────────────

def generate_password(length: int = 10) -> str:
    chars = string.ascii_letters.replace("l", "").replace("I", "").replace("O", "") + string.digits
    return "".join(random.choices(chars, k=length))


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Email via Brevo HTTP API ──────────────────────────────────────

def send_password_email(to_email: str, password: str, is_new_user: bool) -> bool:
    """
    Send via Brevo REST API over HTTPS port 443.
    Works on Render — no SMTP ports needed.
    Set BREVO_API_KEY in environment variables.
    """
    api_key = os.getenv("BREVO_API_KEY", "")
    if not api_key:
        print(f"[Auth] BREVO_API_KEY not set. Password for {to_email}: {password}")
        return False

    subject = "Welcome to Nexus-AI — Your Password" if is_new_user else "Nexus-AI — Your New Password"
    intro   = "Your account has been created." if is_new_user else "A new password has been generated for your account."

    body_html = f"""
<div style="font-family:sans-serif;max-width:480px;margin:auto;padding:32px;background:#0F1117;color:#fff;border-radius:12px;">
  <h2 style="color:#3B82F6;margin-bottom:8px;">{'Welcome to Nexus-AI 🚀' if is_new_user else 'Your New Nexus-AI Password'}</h2>
  <p style="color:#aaa;">{intro}</p>
  <div style="background:#1A1D2E;border:1px solid #2a2d3e;border-radius:8px;padding:20px;margin:24px 0;text-align:center;">
    <p style="color:#aaa;font-size:13px;margin:0 0 8px;">Your password</p>
    <p style="font-size:28px;font-weight:900;letter-spacing:4px;color:#fff;margin:0;">{password}</p>
  </div>
  <p style="color:#aaa;font-size:13px;">Use "Forgot Password" on the sign-in page to get a new one anytime.</p>
</div>
"""

    payload = json.dumps({
        "sender":      {"name": "Nexus-AI", "email": "deepakthillai07@gmail.com"},
        "to":          [{"email": to_email}],
        "subject":     subject,
        "htmlContent": body_html,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=payload,
            headers={
                "accept":       "application/json",
                "api-key":      api_key,
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[Auth] Email sent to {to_email} via Brevo API (HTTP {resp.status})")
            return True
    except urllib.error.HTTPError as e:
        print(f"[Auth] Brevo API error {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"[Auth] Brevo API failed for {to_email}: {e}")
        return False
