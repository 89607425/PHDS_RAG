import os
import secrets
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from database import get_conn

VERIFICATION_CODE_EXPIRE_MINUTES = 5
SESSION_EXPIRE_HOURS = 24


def _generate_code() -> str:
    return str(secrets.randbelow(900000) + 100000)


def _generate_token() -> str:
    return secrets.token_hex(32)


def send_verification_code(email: str) -> str:
    code = _generate_code()
    expires_at = datetime.now() + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES)

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO verification_codes (email, code, expires_at) VALUES (%s, %s, %s)",
            (email, code, expires_at),
        )
    conn.close()

    smtp_host = os.getenv("SMTP_HOST", "")
    if smtp_host:
        try:
            msg = MIMEText(
                f"您的公司知识助手验证码是：{code}\n有效期 {VERIFICATION_CODE_EXPIRE_MINUTES} 分钟。",
                "plain", "utf-8",
            )
            msg["Subject"] = "公司知识助手 — 登录验证码"
            msg["From"] = os.getenv("SMTP_FROM", smtp_host)
            msg["To"] = email
            with smtplib.SMTP_SSL(
                smtp_host, int(os.getenv("SMTP_PORT", "465"))
            ) as server:
                server.login(os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", ""))
                server.send_message(msg)
        except Exception:
            pass

    print(f"\n{'='*50}")
    print(f"  验证码: {code}")
    print(f"  邮箱  : {email}")
    print(f"  有效期: {VERIFICATION_CODE_EXPIRE_MINUTES} 分钟")
    print(f"{'='*50}\n")
    return code


def verify_code(email: str, code: str) -> tuple[bool, str]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, expires_at, used FROM verification_codes WHERE email=%s AND code=%s ORDER BY id DESC LIMIT 1",
            (email, code),
        )
        row = cur.fetchone()
    conn.close()

    if not row:
        return False, "验证码错误"
    if row["used"]:
        return False, "验证码已使用"
    if datetime.now() > row["expires_at"]:
        return False, "验证码已过期"

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("UPDATE verification_codes SET used=1 WHERE id=%s", (row["id"],))
    conn.close()

    return True, ""


def get_or_create_user(email: str) -> int:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        row = cur.fetchone()
        if row:
            user_id = row["id"]
        else:
            cur.execute("INSERT INTO users (email) VALUES (%s)", (email,))
            user_id = cur.lastrowid
    conn.close()
    return user_id


def create_session(email: str) -> str:
    user_id = get_or_create_user(email)
    token = _generate_token()
    expires_at = datetime.now() + timedelta(hours=SESSION_EXPIRE_HOURS)

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (%s, %s, %s)",
            (user_id, token, expires_at),
        )
    conn.close()
    return token


def validate_session(token: str) -> tuple[bool, int | None, str | None]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.user_id, s.expires_at, u.email FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=%s",
            (token,),
        )
        row = cur.fetchone()
    conn.close()

    if not row:
        return False, None, None
    if datetime.now() > row["expires_at"]:
        return False, None, None
    return True, row["user_id"], row["email"]


def get_user_conversations(user_id: int) -> list[dict]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, created_at FROM conversations WHERE user_id=%s ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]


def create_conversation(user_id: int, title: str) -> int:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s)",
            (user_id, title),
        )
        conv_id = cur.lastrowid
    conn.close()
    return conv_id


def delete_conversation(conv_id: int, user_id: int):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM conversations WHERE id=%s AND user_id=%s",
            (conv_id, user_id),
        )
    conn.close()


def get_conversation_messages(conv_id: int) -> list[dict]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id=%s ORDER BY id ASC",
            (conv_id,),
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_message(conv_id: int, role: str, content: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conv_id, role, content),
        )
        cur.execute(
            "UPDATE conversations SET updated_at=NOW() WHERE id=%s", (conv_id,)
        )
    conn.close()


def get_conversation_summary(conv_id: int) -> tuple[str | None, int]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT summary, summarized_until_message_id FROM conversations WHERE id=%s",
            (conv_id,),
        )
        row = cur.fetchone()
    conn.close()
    if row:
        return row.get("summary"), row.get("summarized_until_message_id", 0)
    return None, 0


def update_conversation_summary(conv_id: int, summary: str, until_message_id: int):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE conversations SET summary=%s, summarized_until_message_id=%s WHERE id=%s",
            (summary, until_message_id, conv_id),
        )
    conn.close()


def get_conversation_messages_from(conv_id: int, after_message_id: int = 0) -> list[dict]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, role, content FROM messages WHERE conversation_id=%s AND id > %s ORDER BY id ASC",
            (conv_id, after_message_id),
        )
        rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]
