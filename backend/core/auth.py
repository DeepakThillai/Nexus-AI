"""
auth.py — Password generation, hashing, verification, and email sending.
"""
import os
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import bcrypt

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# ── Password helpers ──────────────────────────────────────────────

def generate_password(length: int = 10) -> str:
    """Generate a readable random password: letters + digits, no ambiguous chars."""
    chars = string.ascii_letters.replace("l", "").replace("I", "").replace("O", "") + string.digits
    return "".join(random.choices(chars, k=length))


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Email sending ─────────────────────────────────────────────────

def _get_smtp_config():
    return {
        "host":     os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port":     int(os.getenv("SMTP_PORT", "587")),
        "user":     os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", "").replace(" ", ""),  # strip spaces from app password
        "from":     os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")),
    }


def send_password_email(to_email: str, password: str, is_new_user: bool) -> bool:
    """
    Send the generated password to the user's email.
    Returns True on success, False on failure (fails silently so auth still works).
    """
    cfg = _get_smtp_config()
    if not cfg["user"] or not cfg["password"]:
        print("[Auth] SMTP not configured — password not emailed. "
              f"Set SMTP_USER and SMTP_PASSWORD in .env. Password: {password}")
        return False

    subject = "Welcome to Nexus-AI — Your Password" if is_new_user else "Nexus-AI — Your New Password"

    if is_new_user:
        body_html = f"""
<div style="font-family:sans-serif;max-width:480px;margin:auto;padding:32px;background:#0F1117;color:#fff;border-radius:12px;">
  <h2 style="color:#3B82F6;margin-bottom:8px;">Welcome to Nexus-AI 🚀</h2>
  <p style="color:#aaa;">Your account has been created. Use the password below to sign in.</p>
  <div style="background:#1A1D2E;border:1px solid #2a2d3e;border-radius:8px;padding:20px;margin:24px 0;text-align:center;">
    <p style="color:#aaa;font-size:13px;margin:0 0 8px;">Your password</p>
    <p style="font-size:28px;font-weight:900;letter-spacing:4px;color:#fff;margin:0;">{password}</p>
  </div>
  <p style="color:#aaa;font-size:13px;">Keep this safe. You can request a new password anytime using "Forgot Password".</p>
</div>
"""
    else:
        body_html = f"""
<div style="font-family:sans-serif;max-width:480px;margin:auto;padding:32px;background:#0F1117;color:#fff;border-radius:12px;">
  <h2 style="color:#3B82F6;margin-bottom:8px;">Your New Nexus-AI Password</h2>
  <p style="color:#aaa;">A new password has been generated for your account.</p>
  <div style="background:#1A1D2E;border:1px solid #2a2d3e;border-radius:8px;padding:20px;margin:24px 0;text-align:center;">
    <p style="color:#aaa;font-size:13px;margin:0 0 8px;">Your new password</p>
    <p style="font-size:28px;font-weight:900;letter-spacing:4px;color:#fff;margin:0;">{password}</p>
  </div>
  <p style="color:#aaa;font-size:13px;">If you did not request this, contact support.</p>
</div>
"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg["from"]
        msg["To"]      = to_email
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["from"], to_email, msg.as_string())

        print(f"[Auth] Password email sent to {to_email}")
        return True

    except Exception as e:
        print(f"[Auth] Failed to send email to {to_email}: {e}")
        return False
