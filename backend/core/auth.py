"""
auth.py — Password generation, hashing, verification, and email sending.
Uses Brevo SMTP relay for email delivery.
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
    chars = string.ascii_letters.replace("l", "").replace("I", "").replace("O", "") + string.digits
    return "".join(random.choices(chars, k=length))


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Email sending via Brevo SMTP ──────────────────────────────────

def send_password_email(to_email: str, password: str, is_new_user: bool) -> bool:
    smtp_host = os.getenv("SMTP_HOST", "smtp-relay.brevo.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "").replace(" ", "")
    from_addr = os.getenv("SMTP_FROM", smtp_user)

    if not smtp_user or not smtp_pass:
        print(f"[Auth] SMTP not configured — password not emailed. Password: {password}")
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
  <p style="color:#aaa;font-size:13px;">Keep this safe. Use "Forgot Password" on the sign-in page to get a new one anytime.</p>
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
        msg["From"]    = f"Nexus-AI <{from_addr}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, to_email, msg.as_string())

        print(f"[Auth] Password email sent to {to_email} via Brevo SMTP")
        return True
    except Exception as e:
        print(f"[Auth] SMTP failed for {to_email}: {e}")
        return False
