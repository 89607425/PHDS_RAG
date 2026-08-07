"""公司知识助手 — FastAPI + 邮箱验证码登录 + MySQL 持久化 + SPA 前端"""
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
from rag_chain import ask, _check_knowledge_base
from database import init_db
from auth import (
    send_verification_code, verify_code, create_session, validate_session,
    get_user_conversations, create_conversation, delete_conversation,
    get_conversation_messages, save_message,
)
from dotenv import load_dotenv

load_dotenv()
init_db()

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>公司知识助手 - 登录</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;background:#f5f5f7;display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:#fff;border-radius:16px;padding:40px 36px;width:380px;box-shadow:0 2px 12px rgba(20,24,35,0.06),0 1px 4px rgba(20,24,35,0.04)}
.card h1{font-size:22px;color:#1c1c1e;text-align:center;margin-bottom:4px}
.card .sub{font-size:13px;color:#9aa0aa;text-align:center;margin-bottom:28px}
.card label{display:block;font-size:13px;color:#5f6470;margin-bottom:6px;font-weight:500}
.card input[type=email],.card input[type=text]{width:100%;padding:10px 14px;border:1px solid #e6e8ec;border-radius:10px;font-size:15px;color:#1c1c1e;outline:none;transition:border-color .2s}
.card input:focus{border-color:#2563eb}
.card .row{display:flex;gap:10px}
.card .row input{flex:1}
.card button{border:none;border-radius:10px;padding:10px 20px;font-size:14px;cursor:pointer;font-weight:500;transition:background .2s,opacity .2s;white-space:nowrap}
.card .btn-pri{background:#2563eb;color:#fff;width:100%;margin-top:20px}
.card .btn-pri:hover{background:#1d4ed8}
.card .btn-pri:disabled{opacity:0.5;cursor:not-allowed}
.card .btn-sec{background:#f1f3f7;color:#2563eb}
.card .btn-sec:hover{background:#e8f1fc}
.card .msg{font-size:12px;margin-top:8px;text-align:center;min-height:18px}
.card .msg.ok{color:#16a34a}
.card .msg.err{color:#dc2626}
.card .step2{display:none}
.card .step2.show{display:block}
</style>
</head>
<body>
<div class="card">
<h1>📋 公司知识助手</h1>
<div class="sub">请输入工作邮箱登录</div>
<div id="step1">
  <label>邮箱地址</label>
  <div class="row">
    <input type="email" id="email" placeholder="name@company.com" autocomplete="email">
    <button class="btn-sec" onclick="sendCode()" id="sendBtn">获取验证码</button>
  </div>
  <div id="msg1" class="msg"></div>
</div>
<div id="step2" class="step2">
  <label>验证码</label>
  <input type="text" id="code" placeholder="6 位验证码" maxlength="6" autocomplete="one-time-code">
  <button class="btn-pri" onclick="verifyLogin()" id="loginBtn">登录</button>
  <div id="msg2" class="msg"></div>
</div>
</div>
<script>
let sentEmail='';let countdown=0;
function setMsg(el,text,cls){document.getElementById(el).textContent=text;document.getElementById(el).className='msg '+cls}
async function sendCode(){
  const email=document.getElementById('email').value.trim();
  if(!email){setMsg('msg1','请输入邮箱','err');return}
  document.getElementById('sendBtn').disabled=true;
  document.getElementById('sendBtn').textContent='发送中...';
  try{
    const r=await fetch('/login/send-code',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email})});
    const d=await r.json();
    if(r.ok){sentEmail=email;document.getElementById('step1').style.display='none';document.getElementById('step2').classList.add('show');setMsg('msg2','验证码已发送（控制台可查看）','ok');countdown=60;updateCountdown()}
    else{setMsg('msg1',d.detail||'发送失败','err');document.getElementById('sendBtn').disabled=false;document.getElementById('sendBtn').textContent='获取验证码'}
  }catch(e){setMsg('msg1','网络错误','err');document.getElementById('sendBtn').disabled=false;document.getElementById('sendBtn').textContent='获取验证码'}
}
function updateCountdown(){
  if(countdown<=0){document.getElementById('sendBtn').disabled=false;document.getElementById('sendBtn').textContent='重新发送';return}
  document.getElementById('sendBtn').textContent=countdown+'s';countdown--;setTimeout(updateCountdown,1000);
}
async function verifyLogin(){
  const code=document.getElementById('code').value.trim();
  if(!code||code.length!==6){setMsg('msg2','请输入6位验证码','err');return}
  document.getElementById('loginBtn').disabled=true;document.getElementById('loginBtn').textContent='验证中...';
  try{
    const r=await fetch('/login/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:sentEmail,code})});
    if(r.ok){window.location.href='/chat'}
    else{const d=await r.json();setMsg('msg2',d.detail||'验证失败','err');document.getElementById('loginBtn').disabled=false;document.getElementById('loginBtn').textContent='登录'}
  }catch(e){setMsg('msg2','网络错误','err');document.getElementById('loginBtn').disabled=false;document.getElementById('loginBtn').textContent='登录'}
}
</script>
</body>
</html>"""


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        allowed = {"/", "/login", "/login/send-code", "/login/verify", "/logout",
                    "/api/health", "/docs", "/openapi.json", "/favicon.ico"}
        if path in allowed or path.startswith("/static/"):
            return await call_next(request)
        if path.startswith("/chat") or path.startswith("/api/"):
            token = request.cookies.get("session_token")
            if not token:
                return RedirectResponse("/login", 302) if not path.startswith("/api/") else JSONResponse({"detail": "未登录"}, 401)
            valid, _, _ = validate_session(token)
            if not valid:
                resp = RedirectResponse("/login", 302) if not path.startswith("/api/") else JSONResponse({"detail": "登录已过期"}, 401)
                resp.delete_cookie("session_token")
                return resp
        return await call_next(request)


app = FastAPI(title="公司知识助手 API", version="2.0.0")
app.add_middleware(AuthMiddleware)

try:
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
except Exception:
    pass


def _get_user_id(request: Request) -> int:
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(401, "未登录")
    valid, uid, _ = validate_session(token)
    if not valid:
        raise HTTPException(401, "登录已过期")
    return uid


def _get_user_email(request: Request) -> tuple:
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(401, "未登录")
    valid, uid, email = validate_session(token)
    if not valid:
        raise HTTPException(401, "登录已过期")
    return uid, email


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return LOGIN_PAGE


@app.post("/login/send-code")
async def login_send_code(req: Request):
    body = await req.json()
    email = body.get("email", "").strip()
    if not email or "@" not in email:
        return JSONResponse({"detail": "请输入有效的邮箱地址"}, 400)
    send_verification_code(email)
    return {"ok": True}


@app.post("/login/verify")
async def login_verify(req: Request):
    body = await req.json()
    email = body.get("email", "").strip()
    code = body.get("code", "").strip()
    ok, err = verify_code(email, code)
    if not ok:
        return JSONResponse({"detail": err}, 400)
    token = create_session(email)
    resp = JSONResponse({"ok": True})
    resp.set_cookie("session_token", token, max_age=86400, httponly=True, samesite="lax", path="/")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", 302)
    resp.delete_cookie("session_token")
    return resp


@app.get("/")
def root(request: Request):
    token = request.cookies.get("session_token")
    logged_in = False
    if token:
        valid, _, _ = validate_session(token)
        if valid:
            logged_in = True
    chat_link = '<a href="/chat" style="color:#2563eb">💬 进入聊天</a>' if logged_in else '<a href="/login" style="color:#2563eb">🔐 登录</a>'
    return HTMLResponse(f'<html><body style="font-family:sans-serif;text-align:center;padding-top:80px;background:#f5f5f7;color:#1c1c1e"><h1>📋 公司知识助手</h1><p>{chat_link} | <a href="/docs" style="color:#2563eb">📖 API 文档</a></p></body></html>')


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    fp = os.path.join(BASE_DIR, "static", "chat.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse("<h1>聊天页面加载失败</h1>", 500)


@app.get("/api/health")
def health():
    return {"status": "ok", "has_kb": _check_knowledge_base()}


@app.get("/api/me")
def api_me(request: Request):
    uid, email = _get_user_email(request)
    return {"user_id": uid, "email": email}


@app.get("/api/conversations")
def api_list_conversations(request: Request):
    uid = _get_user_id(request)
    rows = get_user_conversations(uid)
    result = []
    for r in rows:
        msgs = get_conversation_messages(r["id"])
        result.append({
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"].strftime("%Y-%m-%dT%H:%M:%S") if isinstance(r["created_at"], datetime) else str(r["created_at"]),
            "message_count": len(msgs),
        })
    return result


@app.post("/api/conversations")
async def api_create_conversation(request: Request):
    body = await request.json()
    title = body.get("title", "").strip() or f"对话 {datetime.now().strftime('%H:%M')}"
    uid = _get_user_id(request)
    conv_id = create_conversation(uid, title)
    return {"id": conv_id, "title": title}


@app.delete("/api/conversations/{conv_id}")
def api_delete_conversation(conv_id: int, request: Request):
    uid = _get_user_id(request)
    delete_conversation(conv_id, uid)
    return {"ok": True}


@app.get("/api/conversations/{conv_id}/messages")
def api_get_messages(conv_id: int, request: Request):
    _get_user_id(request)
    msgs = get_conversation_messages(conv_id)
    return msgs


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    uid = _get_user_id(request)
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")

    conv_id = req.conversation_id
    if conv_id is None:
        title = question[:30] + ("..." if len(question) > 30 else "")
        conv_id = create_conversation(uid, title)

    save_message(conv_id, "user", question)
    result = ask(question)
    save_message(conv_id, "assistant", result["answer"])

    return {
        "conversation_id": conv_id,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "has_kb": result.get("has_kb", False),
        "latency_ms": result.get("latency_ms", 0),
        "tokens_used": result.get("tokens_used", 0),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
