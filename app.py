import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import gradio as gr
from rag_chain import ask, _check_knowledge_base
from dotenv import load_dotenv

load_dotenv()

CSS = """
.gradio-container { max-width: 880px !important; margin: auto !important; }
.title { text-align: center; font-size: 1.8em; font-weight: bold; margin-bottom: 0; }
.subtitle { text-align: center; color: #888; margin-top: 4px; margin-bottom: 20px; }
.kb-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.85em;
    font-weight: 500; margin-bottom: 14px;
}
.kb-on  { background: #e8f5e9; color: #2e7d32; }
.kb-off { background: #fff3e0; color: #e65100; }
.source-tag {
    display: inline-block; background: #e8eaf6; color: #283593;
    padding: 2px 10px; border-radius: 12px; font-size: 0.8em; margin: 2px 4px;
}
footer { visibility: hidden; }
#chatbot { border-radius: 12px; }
"""

def get_kb_html():
    if _check_knowledge_base():
        return '<span class="kb-badge kb-on">✅ 知识库已就绪（基于文档检索回答）</span>'
    return '<span class="kb-badge kb-off">⚠️ 知识库尚未建立（基于 AI 通用知识回答）</span>'

def respond(message, chat_history):
    chat_history = chat_history or []
    try:
        result = ask(message)
    except Exception as e:
        response = f"❌ 调用失败：{str(e)}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        return "", chat_history, get_kb_html()

    answer = result["answer"]
    if result["has_kb"] and result["sources"]:
        srcs = "\n".join(f'<span class="source-tag">📎 {s["title"]}</span>' for s in result["sources"])
        answer += f"\n\n<small>━━━━━━━━━━━━━━━━</small>\n<small>📚 参考来源：{srcs}</small>"
    elif not result["has_kb"]:
        answer += "\n\n<small>━━━━━━━━━━━━━━━━</small>\n<small>⚠️ 知识库尚未建立，以上回答基于通用知识</small>"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history, get_kb_html()

def build_gradio():
    with gr.Blocks(title="公司知识助手") as demo:
        gr.HTML('<div class="title">📋 公司内部知识助手</div>')
        gr.HTML('<div class="subtitle">基于 RAG 技术的智能问答系统 · DeepSeek + 硅基流动</div>')
        kb_status = gr.HTML(value=get_kb_html())
        chatbot = gr.Chatbot(elem_id="chatbot", height=480,
                             placeholder="欢迎使用公司知识助手<br>请输入你的问题开始对话...")
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="输入你的问题，如：差旅报销标准是什么？",
                scale=8, container=False, autofocus=True,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
        send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot, kb_status])
        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot, kb_status])
    return demo

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    has_kb: bool

app = FastAPI(title="公司知识助手 API", description="基于 RAG 的企业内部知识库问答系统", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><body style="font-family:sans-serif;text-align:center;padding-top:80px">
    <h1>📋 公司知识助手</h1>
    <p>API 已就绪 | <a href="/docs">📖 API 文档</a> | <a href="/chat">💬 聊天界面</a></p>
    </body></html>
    """

@app.post("/api/ask", response_model=AskResponse, summary="RAG 知识问答",
          description="提交问题，返回基于知识库的 AI 回答及其参考来源")
def api_ask(req: AskRequest):
    result = ask(req.question)
    return AskResponse(**result)

@app.get("/api/health", summary="健康检查")
def health():
    return {"status": "ok", "has_kb": _check_knowledge_base()}

demo = build_gradio()
app = gr.mount_gradio_app(app, demo, path="/chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
