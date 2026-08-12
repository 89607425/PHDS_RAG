"""公司知识助手 — FastAPI + 邮箱验证码登录 + MySQL 持久化 + SPA 前端"""
import html
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, Form, File
from pydantic import BaseModel
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
import threading
from rag_chain import ask, ask_stream, _check_knowledge_base, maybe_update_summary
from kb_manager import list_documents, upload_document, delete_document, rebuild_index
from database import init_db
from auth import (
    send_verification_code, verify_code, create_session, validate_session,
    get_user_conversations, create_conversation, delete_conversation,
    get_conversation_messages, save_message,
    get_conversation_summary,
)
from dotenv import load_dotenv


load_dotenv()
init_db()

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGIN_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>公司知识助手</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body {
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Helvetica Neue",Helvetica,Arial,sans-serif;
  background:#f0f2f5; display:flex; align-items:center; justify-content:center; min-height:100vh;
  -webkit-font-smoothing:antialiased;
}
.container { width:100%; max-width:440px; padding:24px }
.brand { text-align:center; margin-bottom:36px }
.brand-icon {
  width:64px; height:64px; border-radius:18px;
  background:linear-gradient(135deg,#2563eb,#7c3aed);
  display:inline-flex; align-items:center; justify-content:center;
  font-size:28px; margin-bottom:16px;
  box-shadow:0 8px 24px rgba(37,99,235,0.25);
}
.brand h1 { font-size:24px; font-weight:700; color:#1e293b; letter-spacing:-0.3px }
.brand p { font-size:14px; color:#94a3b8; margin-top:6px }
.card {
  background:#fff; border-radius:20px; padding:32px 28px;
  box-shadow:0 1px 3px rgba(0,0,0,0.04),0 8px 32px rgba(0,0,0,0.06);
}
.step { transition: opacity .25s,transform .25s }
.step.hidden { display:none }
.input-group { margin-bottom:18px }
.input-group label { display:block; font-size:13px; font-weight:600; color:#475569; margin-bottom:6px; letter-spacing:0.2px }
.input-group .hint { font-size:11px; color:#94a3b8; font-weight:400; margin-left:4px }
.input-row { display:flex; gap:10px }
.input-row input { flex:1 }
input[type=email],input[type=text] {
  width:100%; padding:12px 14px; border:1.5px solid #e2e8f0; border-radius:12px;
  font-size:15px; color:#1e293b; outline:none; transition:all .2s; background:#f8fafc;
  font-family:inherit;
}
input:focus { border-color:#2563eb; background:#fff; box-shadow:0 0 0 3px rgba(37,99,235,0.08) }
input::placeholder { color:#cbd5e1 }
.btn {
  border:none; border-radius:12px; padding:11px 20px; font-size:14px; cursor:pointer;
  font-weight:600; transition:all .2s; white-space:nowrap; font-family:inherit;
  display:inline-flex; align-items:center; justify-content:center; gap:6px;
}
.btn-primary { background:linear-gradient(135deg,#2563eb,#4f46e5); color:#fff; width:100%; margin-top:4px }
.btn-primary:hover { transform:translateY(-1px); box-shadow:0 4px 14px rgba(37,99,235,0.35) }
.btn-primary:disabled { opacity:0.5; cursor:not-allowed; transform:none; box-shadow:none }
.btn-ghost { background:transparent; color:#64748b; font-weight:500 }
.btn-ghost:hover { color:#2563eb; background:rgba(37,99,235,0.06) }
.msg { font-size:12px; margin-top:8px; min-height:18px; text-align:center; font-weight:500 }
.msg.ok { color:#059669 }
.msg.err { color:#dc2626 }
.footer { text-align:center; margin-top:28px; font-size:12px; color:#94a3b8; line-height:1.8 }
.footer a { color:#64748b; text-decoration:none; transition:color .2s }
.footer a:hover { color:#2563eb }
.code-box {
  display:flex; gap:10px; justify-content:center; margin:4px 0 18px
}
.code-digit {
  width:44px; height:52px; border:2px solid #e2e8f0; border-radius:12px;
  font-size:20px; text-align:center; font-family:"SF Mono","Monaco","Menlo",monospace;
  font-weight:600; color:#1e293b; outline:none; background:#f8fafc;
  transition:all .2s;
}
.code-digit:focus { border-color:#2563eb; background:#fff; box-shadow:0 0 0 3px rgba(37,99,235,0.08) }
.spinner { width:18px; height:18px; border:2px solid rgba(255,255,255,0.3); border-top-color:#fff; border-radius:50%; animation:spin .6s linear infinite; display:none }
@keyframes spin { to { transform:rotate(360deg) } }
.powered { display:flex; align-items:center; justify-content:center; gap:6px; margin-top:20px; font-size:11px; color:#94a3b8 }
.powered-dot {
  width:10px; height:10px; border-radius:3px; background:linear-gradient(135deg,#2563eb,#7c3aed)
}
</style>
</head>
<body>
<div class="container">
<div class="brand">
  <div class="brand-icon">📋</div>
  <h1>公司知识助手</h1>
  <p>企业级 RAG 智能问答系统</p>
</div>
<div class="card">
  <div class="step" id="stepEmail">
    <div class="input-group">
      <label>工作邮箱</label>
      <input type="email" id="email" placeholder="name@company.com" autocomplete="email" autofocus>
    </div>
    <button class="btn btn-primary" onclick="sendCode()" id="sendBtn">
      <span id="sendBtnText">发送验证码</span>
      <span class="spinner" id="sendSpinner"></span>
    </button>
    <div id="msgEmail" class="msg"></div>
  </div>
  <div class="step hidden" id="stepCode">
    <div style="text-align:center;margin-bottom:6px">
      <span style="font-size:13px;color:#64748b">验证码已发送至</span>
      <span style="font-size:13px;color:#1e293b;font-weight:600" id="sentEmailDisplay"></span>
    </div>
    <div class="code-box" id="codeInputs">
      <input type="text" class="code-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" oninput="onCodeInput(this)" onkeydown="onCodeKeydown(event,0)" onpaste="onCodePaste(event)">
      <input type="text" class="code-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" oninput="onCodeInput(this)" onkeydown="onCodeKeydown(event,1)" onpaste="onCodePaste(event)">
      <input type="text" class="code-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" oninput="onCodeInput(this)" onkeydown="onCodeKeydown(event,2)" onpaste="onCodePaste(event)">
      <input type="text" class="code-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" oninput="onCodeInput(this)" onkeydown="onCodeKeydown(event,3)" onpaste="onCodePaste(event)">
      <input type="text" class="code-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" oninput="onCodeInput(this)" onkeydown="onCodeKeydown(event,4)" onpaste="onCodePaste(event)">
      <input type="text" class="code-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" oninput="onCodeInput(this)" onkeydown="onCodeKeydown(event,5)" onpaste="onCodePaste(event)">
    </div>
    <div id="msgCode" class="msg"></div>
    <button class="btn btn-primary" onclick="verifyLogin()" id="loginBtn" style="margin-top:20px">
      <span id="loginBtnText">验证并登录</span>
      <span class="spinner" id="loginSpinner"></span>
    </button>
    <div style="text-align:center;margin-top:16px">
      <button class="btn btn-ghost" onclick="goBack()" style="font-size:13px">← 更换邮箱</button>
      <span style="margin:0 8px;color:#e2e8f0">|</span>
      <button class="btn btn-ghost" onclick="sendCode()" id="resendBtn" style="font-size:13px" disabled>重新发送 (60s)</button>
    </div>
  </div>
</div>

</div>
<script>
var sentEmail='';var countdown=0;var timers=[];
function setMsg(id,text,cls){var el=document.getElementById(id);el.textContent=text;el.className='msg '+cls}
function showStep(s){var steps={email:'stepEmail',code:'stepCode'};for(var k in steps)document.getElementById(steps[k]).classList[k===s?'remove':'add']('hidden')}
function getCode(){var s='';for(var i=0;i<6;i++){s+=document.getElementById('codeInputs').children[i].value}return s}
function focusCode(i){var inputs=document.getElementById('codeInputs').children;if(i>=0&&i<6)inputs[i].focus()}
function onCodeInput(el){el.value=el.value.replace(/\D/g,'');var idx=Array.from(el.parentElement.children).indexOf(el);if(el.value&&idx<5)focusCode(idx+1);checkAutoLogin()}
function onCodeKeydown(e,idx){if(e.key==='Backspace'&&!e.target.value&&idx>0)focusCode(idx-1)}
function onCodePaste(e){e.preventDefault();var text=(e.clipboardData||window.clipboardData).getData('text').replace(/\D/g,'');var inputs=document.getElementById('codeInputs').children;for(var i=0;i<6&&i<text.length;i++){inputs[i].value=text[i]}if(text.length>0)focusCode(Math.min(text.length,5));checkAutoLogin()}
function checkAutoLogin(){if(getCode().length===6)verifyLogin()}
function goBack(){showStep('email');setMsg('msgCode','','');document.getElementById('email').focus()}
function setLoading(btnId,btnText,spinnerId,loading){var btn=document.getElementById(btnId);var text=document.getElementById(btnText);var sp=document.getElementById(spinnerId);btn.disabled=loading;sp.style.display=loading?'inline-block':'none';text.style.opacity=loading?'0':'1'}
async function sendCode(){
  var email=document.getElementById('email').value.trim();
  if(!email||email.indexOf('@')<0){setMsg('msgEmail','请输入有效的邮箱地址','err');return}
  setLoading('sendBtn','sendBtnText','sendSpinner',true);
  try{
    var r=await fetch('/login/send-code',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:email})});
    var d=await r.json();
    if(r.ok){
      sentEmail=email;document.getElementById('sentEmailDisplay').textContent=email;
      showStep('code');setMsg('msgEmail','','');setLoading('sendBtn','sendBtnText','sendSpinner',false);
      countdown=60;updateCountdown;updateResend();focusCode(0);
    }else{
      setMsg('msgEmail',d.detail||'发送失败','err');setLoading('sendBtn','sendBtnText','sendSpinner',false)
    }
  }catch(e){setMsg('msgEmail','网络错误，请重试','err');setLoading('sendBtn','sendBtnText','sendSpinner',false)}
}
function updateResend(){
  var btn=document.getElementById('resendBtn');
  if(countdown<=0){btn.disabled=false;btn.textContent='重新发送';return}
  btn.disabled=true;btn.textContent='重新发送 ('+countdown+'s)';countdown--;
  timers.push(setTimeout(updateResend,1000))
}
async function verifyLogin(){
  var code=getCode();
  if(code.length!==6){setMsg('msgCode','请输入完整的6位验证码','err');return}
  setLoading('loginBtn','loginBtnText','loginSpinner',true);
  try{
    var r=await fetch('/login/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:sentEmail,code:code})});
    if(r.ok){window.location.href='/chat'}
    else{var d=await r.json();setMsg('msgCode',d.detail||'验证失败，请重试','err');setLoading('loginBtn','loginBtnText','loginSpinner',false)}
  }catch(e){setMsg('msgCode','网络错误，请重试','err');setLoading('loginBtn','loginBtnText','loginSpinner',false)}
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
    if token:
        valid, _, _ = validate_session(token)
        if valid:
            return RedirectResponse("/chat", 302)
    return HTMLResponse(LOGIN_PAGE)


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


def _strip_html(text: str) -> str:
    import re
    return re.sub(r'<[^>]+>', '', text)


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    image: str | None = None


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

    summary, summarized_until = get_conversation_summary(conv_id)
    history_msgs = get_conversation_messages(conv_id)
    history = [{"role": m["role"], "content": _strip_html(m["content"])} for m in history_msgs[:-1]]
    result = ask(question, conversation_history=history if history else None,
                 conv_id=conv_id, summary=summary, summarized_until=summarized_until,
                 image_base64=req.image)

    sources = result.get("sources", [])
    stored_content = result["answer"]
    if sources:
        src_items = ""
        for i, s in enumerate(sources, 1):
            rerank_info = f" | rerank={s['rerank_score']:.3f}" if s.get("rerank_score") else ""
            src_items += (
                f'<div class="source-item">'
                f'<div class="source-item-title">来源 {i}: {html.escape(s["title"])} (chunk #{s["chunk_index"]}{rerank_info})</div>'
                f'<div class="source-item-text">{html.escape(s["content"])}</div>'
                f'</div>'
            )
        stored_content += (
            f'\n\n<details class="msg-sources">'
            f'<summary>📚 检索到的知识来源 (Top-{len(sources)})</summary>'
            f'{src_items}</details>'
        )
    save_message(conv_id, "assistant", stored_content)
    maybe_update_summary(conv_id)

    return {
        "conversation_id": conv_id,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "has_kb": result.get("has_kb", False),
        "latency_ms": result.get("latency_ms", 0),
        "tokens_used": result.get("tokens_used", 0),
        "self_check_score": result.get("self_check_score"),
    }


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest, request: Request):
    uid = _get_user_id(request)
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")

    conv_id = req.conversation_id
    if conv_id is None:
        title = question[:30] + ("..." if len(question) > 30 else "")
        conv_id = create_conversation(uid, title)

    save_message(conv_id, "user", question)

    summary, summarized_until = get_conversation_summary(conv_id)
    history_msgs = get_conversation_messages(conv_id)
    history = [{"role": m["role"], "content": _strip_html(m["content"])} for m in history_msgs[:-1]]

    def generate():
        full_answer = ""
        sources = []
        conv_done = {"conv_id": conv_id}
        try:
            for line in ask_stream(question, conversation_history=history if history else None,
                                      conv_id=conv_id, summary=summary, summarized_until=summarized_until,
                                      image_base64=req.image):
                try:
                    event = json.loads(line)
                    if event["type"] == "sources":
                        sources = event["data"]
                    elif event["type"] == "token":
                        full_answer += event["data"]
                    yield f"data: {line}\n"
                except json.JSONDecodeError:
                    yield f"data: {line}\n"

            yield f"data: {json.dumps({'type': 'done', 'data': conv_done}, ensure_ascii=False)}\n\n"

            def save_after():
                stored = full_answer
                if sources:
                    src_items = ""
                    for i, s in enumerate(sources, 1):
                        rerank_info = f" | rerank={s['rerank_score']:.3f}" if s.get("rerank_score") else ""
                        src_items += (
                            f'<div class="source-item">'
                            f'<div class="source-item-title">来源 {i}: {html.escape(s["title"])} (chunk #{s["chunk_index"]}{rerank_info})</div>'
                            f'<div class="source-item-text">{html.escape(s["content"])}</div>'
                            f'</div>'
                        )
                    stored += (
                        f'\n\n<details class="msg-sources">'
                        f'<summary>📚 检索到的知识来源 (Top-{len(sources)})</summary>'
                        f'{src_items}</details>'
                    )
                save_message(conv_id, "assistant", stored)
                maybe_update_summary(conv_id)
            threading.Thread(target=save_after, daemon=True).start()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/kb/list")
def api_kb_list(request: Request):
    _get_user_id(request)
    docs = list_documents()
    for d in docs:
        d["modified"] = datetime.fromtimestamp(d["modified"]).strftime("%Y-%m-%dT%H:%M:%S")
    return docs


@app.post("/api/kb/upload")
async def api_kb_upload(request: Request, file: UploadFile = File(...)):
    _get_user_id(request)
    if not file.filename:
        raise HTTPException(400, "未选择文件")
    content = await file.read()
    result = upload_document(content, file.filename)
    return JSONResponse(result)


@app.delete("/api/kb/delete")
async def api_kb_delete(request: Request):
    body = await request.json()
    filename = body.get("filename", "").strip()
    if not filename:
        raise HTTPException(400, "文件名不能为空")
    ok = delete_document(filename)
    if not ok:
        raise HTTPException(404, f"文件不存在: {filename}")
    return {"ok": True}


@app.post("/api/kb/rebuild")
def api_kb_rebuild(request: Request):
    _get_user_id(request)
    result = rebuild_index()
    return JSONResponse(result)


@app.get("/kb", response_class=HTMLResponse)
def kb_page():
    fp = os.path.join(BASE_DIR, "static", "kb.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse("<h1>知识库管理页面加载失败</h1>", 500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
