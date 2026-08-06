"""RAG 知识问答前端 — Gradio + FastAPI

本文件只在前端样式与布局上做了重构，RAG 检索逻辑位于 rag_chain.py，未做任何改动。
"""
import base64
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from rag_chain import ask, _check_knowledge_base
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# 内嵌 SVG 头像(避免外链失败),可直接用于 Chatbot(avatar_images=...)
# ---------------------------------------------------------------------------
def _svg_avatar(grad: tuple[str, str]) -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{grad[0]}"/>'
        f'<stop offset="100%" stop-color="{grad[1]}"/>'
        '</linearGradient></defs>'
        '<rect width="32" height="32" rx="9" fill="url(#g)"/>'
        '</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _svg_user_avatar() -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="9" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>'
        '<circle cx="16" cy="13" r="4.2" fill="#9ca3af"/>'
        '<path d="M6 26c2.2-4.2 6-6.2 10-6.2s7.8 2 10 6.2" fill="#9ca3af"/>'
        '</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


BOT_AVATAR = _svg_avatar(("#fbbf77", "#fdba74"))   # 暖橙色渐变
USER_AVATAR = _svg_user_avatar()                   # 白色描边、灰头像剪影


# ---------------------------------------------------------------------------
# 浅色主题样式 — 重点:不与 Gradio 6 的 flex 布局打架
# ---------------------------------------------------------------------------
CSS = """
/* ===== 设计变量 ===== */
:root {
    --bg-page:        #f5f5f7;
    --bg-sidebar:    #ffffff;
    --bg-bubble-user:#e8f1fc;
    --bg-card:        #ffffff;
    --bg-hover:        #f1f3f7;
    --bg-active:        #e8f1fc;
    --border:        #e6e8ec;
    --border-soft:    #eef0f3;
    --text-primary:    #1c1c1e;
    --text-secondary:#5f6470;
    --text-muted:    #9aa0aa;
    --accent:        #2563eb;
    --accent-strong:#1d4ed8;
    --accent-soft:    #d6e4ff;
    --shadow-soft: 0 1px 2px rgba(20, 24, 35, 0.04);
    --shadow-card: 0 1px 3px rgba(20, 24, 35, 0.06), 0 1px 2px rgba(20, 24, 35, 0.04);
}

/* ===== 全局重置 ===== */
html, body, .gradio-container, .gradio-container > .main,
.gradio-container .wrap {
    background: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue",
                 Helvetica, Arial, sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer, .gradio-footer { display: none !important; }

/* ===== 顶层布局:整页铺满 ===== */
.app-shell {
    min-height: 100vh !important;
    width: 100%;
    background: var(--bg-page);
}

/* 让 Gradio 的 row(我用来做左右分栏)正确显示为 flex;列与列之间无空隙 */
.app-row {
    gap: 0 !important;
    align-items: stretch !important;
    background: var(--bg-page);
}
.app-row > .gradio-column,
.app-row > [class*="column"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ===== 侧栏 ===== */
.sidebar {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    padding: 16px 14px 14px 14px !important;
    min-width: 260px !important;
    width: 260px !important;
}

/* 侧栏内部 header(用 gr.HTML 渲染) */
.sb-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 6px 14px 6px;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
}
.sb-header .title { display: flex; align-items: center; gap: 8px; }
.sb-header .title-icon {
    width: 22px; height: 22px;
    border-radius: 6px;
    background: linear-gradient(135deg, #fbbf77, #fdba74);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 12px;
}
.sb-header .toggle {
    color: var(--text-muted); cursor: pointer; font-size: 16px;
    padding: 4px 6px; border-radius: 6px;
}
.sb-header .toggle:hover { background: var(--bg-hover); }

.sb-section-title {
    font-size: 11px; color: var(--text-muted);
    padding: 14px 8px 6px 8px;
    letter-spacing: 0.4px; text-transform: uppercase;
}

/* === 侧栏按钮 === */
.sidebar button.new-chat-btn,
.new-chat-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow-soft) !important;
}
.new-chat-btn:hover { background: var(--bg-active) !important; }

.sidebar .del-btn,
.del-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    margin-top: 10px !important;
}
.del-btn:hover { background: var(--bg-hover) !important; color: #dc2626 !important; }

/* === Radio 列表样式 === */
.conv-scroll .block,
.conv-scroll .gradio-radio {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.conv-scroll label {
    padding: 9px 12px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
    margin: 3px 0 !important;
    transition: background .15s, color .15s;
    width: 100% !important;
}
.conv-scroll label:hover { background: var(--bg-hover) !important; color: var(--text-primary) !important; }
.conv-scroll input:checked + span,
.conv-scroll .selected { color: var(--accent) !important; font-weight: 500 !important; }
.conv-scroll li.selected label,
.conv-scroll li:has(input:checked) label {
    background: var(--bg-active) !important;
    color: var(--accent) !important;
}

.sidebar-kb {
    border-top: 1px solid var(--border-soft);
    padding-top: 12px;
    margin-top: 14px;
    font-size: 11px; color: var(--text-muted);
    line-height: 1.7;
    padding-left: 6px;
}
.sidebar-kb .ok   { color: #16a34a; }
.sidebar-kb .warn { color: #d97706; }

/* ===== 聊天区 ===== */
.chat-col {
    background: var(--bg-page) !important;
    padding: 0 !important;
}
.chat-topbar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 24px 4px 24px;
    color: var(--text-muted);
}
.chat-topbar .icon-btn {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 16px;
}
.chat-topbar .icon-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

.chatbot-shell {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.chatbot-shell > .block,
.chatbot-shell .wrap,
.chatbot-shell .chatbot {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#chatbot, #chatbot .chatbot, #chatbot > div {
    background: transparent !important;
    border: none !important;
}
#chatbot .message,
#chatbot .bubble-wrap,
#chatbot .message-wrap,
#chatbot .message-bubble,
#chatbot [class*="message"],
#chatbot [class*="bubble"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#chatbot img.avatar,
#chatbot .avatar img,
#chatbot .avatar-image img {
    width: 32px !important; height: 32px !important;
    border-radius: 9px !important; object-fit: cover !important;
}
#chatbot .avatar, #chatbot .avatar-image {
    width: 32px !important; height: 32px !important; flex-shrink: 0 !important;
}

/* ===== Source box(检索到的片段) ===== */
.source-box {
    margin-top: 10px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 13.5px !important;
    box-shadow: var(--shadow-soft) !important;
}
.source-box summary {
    cursor: pointer;
    color: var(--text-primary) !important;
    font-weight: 500;
    display: flex; align-items: center; gap: 6px;
    list-style: none;
}
.source-box summary::-webkit-details-marker { display: none; }
.source-box summary::before {
    content: "▶";
    color: var(--accent);
    font-size: 9px;
    transition: transform .2s;
    display: inline-block;
    margin-right: 2px;
}
.source-box[open] summary::before { transform: rotate(90deg); }
.source-item {
    background: var(--bg-hover) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    margin: 8px 0 !important;
    border-left: 3px solid var(--accent) !important;
}
.source-item .src-title { font-weight: 600 !important; color: var(--text-primary) !important; font-size: 13px !important; }
.source-item .src-content { color: var(--text-secondary) !important; font-size: 12.5px !important; margin-top: 6px !important; line-height: 1.6 !important; white-space: pre-wrap; }
.source-item .src-idx { color: var(--text-muted) !important; font-size: 11px !important; }
.msg-meta {
    font-size: 11.5px !important;
    color: var(--text-muted) !important;
    margin-top: 8px !important;
}

/* ===== 输入区(胶囊) ===== */
.input-shell {
    flex: 0 0 auto;
    padding: 8px 24px 6px 24px;
}
.input-card {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 28px !important;
    padding: 6px 8px 6px 20px !important;
    box-shadow: var(--shadow-card) !important;
}
.input-card > .block,
.input-card > .wrap,
.input-card .gradio-textbox,
.input-card .gradio-textbox > .wrap,
.input-card textarea,
.input-card input[type="text"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}
.input-card textarea::placeholder,
.input-card input::placeholder {
    color: var(--text-muted) !important;
    opacity: 0.85 !important;
}

.attach-btn {
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    width: 36px !important; min-width: 36px !important; height: 36px !important;
    border-radius: 50% !important;
    font-size: 18px !important;
    box-shadow: none !important;
}
.attach-btn:hover { background: var(--bg-hover) !important; color: var(--text-secondary) !important; }

.send-btn {
    background: var(--accent) !important;
    border: none !important;
    width: 36px !important; min-width: 36px !important; height: 36px !important;
    border-radius: 50% !important;
    color: #fff !important;
    font-size: 14px !important;
    box-shadow: none !important;
}
.send-btn:hover { background: var(--accent-strong) !important; }

.chat-footer {
    flex: 0 0 auto;
    padding: 6px 24px 14px 28px;
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: var(--text-muted);
}
.chat-footer .brand-avatar {
    width: 18px; height: 18px; border-radius: 5px;
    background: linear-gradient(135deg, #fbbf77, #fdba74);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 10px;
}
"""


# ---------------------------------------------------------------------------
# 回答内容渲染(保留全部原有文案,只重排样式)
# ---------------------------------------------------------------------------
HTML_WRAPPER = (
    '<div class="bot-bubble" '
    'style="font-family:-apple-system,BlinkMacSystemFont,'
    '&quot;PingFang SC&quot;,&quot;Hiragino Sans GB&quot;,'
    '&quot;Microsoft YaHei&quot;,sans-serif;font-size:15px;'
    'line-height:1.65;color:#1c1c1e;white-space:pre-wrap;">{}</div>'
)


def format_answer_html(answer: str, sources: list, latency_ms: int, tokens_used: int, has_kb: bool) -> str:
    parts = [HTML_WRAPPER.format(answer)]
    if has_kb and sources:
        src_items = ""
        for i, s in enumerate(sources, 1):
            src_items += (
                f'<div class="source-item">'
                f'<div class="src-title">📎 来源 {i}:{s["title"]} '
                f'<span class="src-idx">(chunk #{s["chunk_index"]})</span></div>'
                f'<div class="src-content">{s["content"]}</div>'
                f'</div>'
            )
        parts.append(
            f'<details class="source-box">'
            f'<summary>📚 检索到的知识库片段 (Top-5)</summary>'
            f'{src_items}</details>'
        )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latency_s = latency_ms / 1000
    parts.append(
        f'<div class="msg-meta">⏱ {latency_s:.1f}s · {tokens_used} tokens · 📅 {now}</div>'
    )
    return "".join(parts)


def get_kb_html() -> str:
    if _check_knowledge_base():
        return '<span class="ok">✅ 知识库已就绪</span>'
    return '<span class="warn">⚠️ 知识库未建立</span>'


# ---------------------------------------------------------------------------
# 对话状态管理(逻辑保持原状,只换前端样式)
# ---------------------------------------------------------------------------
def init_state():
    return {"conversations": {}, "active_id": ""}


def _build_radio_choices(state):
    if not state["conversations"]:
        return gr.update(choices=[], value=None)
    choices = []
    active_val = None
    for cid, cdata in state["conversations"].items():
        label = f"{cdata['title']} ({cdata['created_at']})"
        choices.append(label)
        if cid == state["active_id"]:
            active_val = label
    return gr.update(choices=choices, value=active_val)


def handle_new(state):
    cid = datetime.now().strftime("%H%M%S")
    title = f"对话 {cid}"
    state["conversations"][cid] = {
        "title": title, "messages": [], "created_at": datetime.now().strftime("%m-%d %H:%M")
    }
    state["active_id"] = cid
    return state, [], _build_radio_choices(state)


def handle_switch(selected, state):
    if not selected or not state["conversations"]:
        return state, [], _build_radio_choices(state)
    for cid, cdata in state["conversations"].items():
        if f"{cdata['title']} ({cdata['created_at']})" == selected:
            state["active_id"] = cid
            return state, cdata["messages"], _build_radio_choices(state)
    return state, [], _build_radio_choices(state)


def handle_delete(state):
    aid = state["active_id"]
    if aid in state["conversations"]:
        del state["conversations"][aid]
    if state["conversations"]:
        state["active_id"] = list(state["conversations"].keys())[-1]
        msgs = state["conversations"][state["active_id"]]["messages"]
    else:
        state["active_id"] = ""
        msgs = []
    return state, msgs, _build_radio_choices(state)


def handle_send(message, chat_history, state):
    if not message or not message.strip():
        return "", chat_history, state, _build_radio_choices(state)

    aid = state["active_id"]
    if not aid:
        state, chat_history, _ = handle_new(state)
        aid = state["active_id"]

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    try:
        result = ask(message)
        html = format_answer_html(
            result["answer"], result.get("sources", []),
            result.get("latency_ms", 0), result.get("tokens_used", 0),
            result.get("has_kb", False),
        )
        chat_history.append({"role": "assistant", "content": html})
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"❌ 调用失败:{str(e)}"})

    state["conversations"][aid]["messages"] = chat_history
    return "", chat_history, state, _build_radio_choices(state)


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------
def build_gradio():
    with gr.Blocks(title="公司知识助手", elem_classes="app-shell") as demo:
        state = gr.State(init_state())

        with gr.Row(elem_classes="app-row"):
            # ---------- 左侧栏 ----------
            with gr.Column(elem_classes="sidebar"):
                gr.HTML(
                    '<div class="sb-header">'
                    '<div class="title">'
                    '<span class="title-icon">📋</span>'
                    '<span>公司知识助手</span>'
                    '</div>'
                    '<div class="toggle" title="折叠">⇤</div>'
                    '</div>'
                )
                new_btn = gr.Button("✨ 开启新对话", elem_classes="new-chat-btn")
                gr.HTML('<div class="sb-section-title">历史对话</div>')
                conv_radio = gr.Radio(
                    choices=[], label="", interactive=True,
                    elem_classes="conv-scroll", show_label=False,
                )
                del_btn = gr.Button("🗑 删除当前对话", elem_classes="del-btn")
                gr.HTML(
                    f'<div class="sidebar-kb">'
                    f'{get_kb_html()}<br/>'
                    'DeepSeek + 硅基流动<br/>'
                    '14 篇 · 707 chunks'
                    f'</div>'
                )

            # ---------- 主聊天区 ----------
            with gr.Column(elem_classes="chat-col", scale=4):
                # 顶部刷新按钮(右对齐)
                gr.HTML(
                    '<div class="chat-topbar">'
                    '<div></div>'
                    '<div class="icon-btn" title="刷新对话">↻</div>'
                    '</div>'
                )

                # 聊天面板(占满剩余高度)
                with gr.Column(elem_classes="chatbot-shell"):
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        avatar_images=(USER_AVATAR, BOT_AVATAR),
                        resizable=True,
                        placeholder="欢迎使用公司知识助手，输入问题开始对话...",
                        show_label=False,
                    )

                # 输入区(胶囊式)
                with gr.Row(elem_classes="input-card"):
                    msg_input = gr.Textbox(
                        placeholder="输入问题，如：客单价怎么算？",
                        scale=8,
                        container=False,
                        autofocus=True,
                        show_label=False,
                    )
                    gr.Button("📎", elem_classes="attach-btn", scale=0)
                    send_btn = gr.Button("➤", elem_classes="send-btn", scale=0)

                # 底部脚注
                gr.HTML(
                    '<div class="chat-footer">'
                    '<span class="brand-avatar">📋</span>'
                    '<span>本系统由 DeepSeek · 硅基流动 强力驱动</span>'
                    '</div>'
                )

        # ---------- 事件绑定 ----------
        send_btn.click(
            handle_send,
            [msg_input, chatbot, state],
            [msg_input, chatbot, state, conv_radio],
        )
        msg_input.submit(
            handle_send,
            [msg_input, chatbot, state],
            [msg_input, chatbot, state, conv_radio],
        )
        new_btn.click(handle_new, state, [state, chatbot, conv_radio])
        conv_radio.change(handle_switch, [conv_radio, state], [state, chatbot, conv_radio])
        del_btn.click(handle_delete, state, [state, chatbot, conv_radio])

    return demo


# ---------------------------------------------------------------------------
# FastAPI 包装(原样保留 API 接口)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    has_kb: bool
    latency_ms: int
    tokens_used: int


app = FastAPI(title="公司知识助手 API", description="基于 RAG 的企业内部知识库问答系统", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def root():
    return (
        '<html><body style="font-family:sans-serif;text-align:center;'
        'padding-top:80px;background:#f5f5f7;color:#1c1c1e">'
        '<h1>📋 公司知识助手</h1>'
        '<p>API 已就绪 | <a href="/docs" style="color:#2563eb">📖 API 文档</a> | '
        '<a href="/chat" style="color:#2563eb">💬 聊天界面</a></p>'
        '</body></html>'
    )


@app.post("/api/ask", response_model=AskResponse, summary="RAG 知识问答")
def api_ask(req: AskRequest):
    result = ask(req.question)
    return AskResponse(**result)


@app.get("/api/health", summary="健康检查")
def health():
    return {"status": "ok", "has_kb": _check_knowledge_base()}


demo = build_gradio()
app = gr.mount_gradio_app(app, demo, path="/chat", css=CSS)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
