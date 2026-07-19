import json
import logging
from fastapi import APIRouter, Header, HTTPException
from semantic_kernel.connectors.ai.open_ai.exceptions.content_filter_ai_exception import ContentFilterAIException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from opentelemetry import trace
from app.agent import build_agent
from app.auth import User, get_user
from app.session import get_session

_tracer = trace.get_tracer("nimbus.chat")

router = APIRouter()

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

def _set_output(span, root, text: str) -> None:
    span.set_attribute("langfuse.observation.output", text)
    if root is not None:
        root.set_attribute("langfuse.trace.output", text)

async def _stream(request: ChatRequest, user: User, root=None):
    with _tracer.start_as_current_span("chat.turn") as span:
        span.set_attribute("conversation.id", request.conversation_id)
        span.set_attribute("user.role", getattr(user, "role", "anonymous"))
        span.set_attribute("langfuse.observation.input", request.message)
        answer_parts: list[str] = []

        session = get_session(request.conversation_id)
        session.user = user
        session.start_turn()
        agent = build_agent(session)

        emitted_tool_calls = 0
        try:
            async for chunk in agent.invoke_stream(messages=request.message, thread=session.thread):
                while emitted_tool_calls < len(session.tool_calls):
                    yield _sse("tool_call", session.tool_calls[emitted_tool_calls])
                    emitted_tool_calls += 1
                session.thread = chunk.thread
                text = str(chunk.content)
                if text:
                    answer_parts.append(text)
                    yield _sse("token", {"text": text})
        except ContentFilterAIException:
            span.set_attribute("outcome", "content_filtered")
            _set_output(span, root, "[blocked by content filter]")
            logging.getLogger("uvicorn.error").warning("Content filter triggered for user %s", user.username)
            yield _sse("error", {"message": "I can't help with that request."})
            return
        except Exception:
            span.set_attribute("outcome", "error")
            _set_output(span, root, "[error]")
            logging.getLogger("uvicorn.error").exception("chat stream failed")
            yield _sse("error", {"message": "An error occurred while processing your request."})
            return
        
        span.set_attribute("outcome", "ok")
        span.set_attribute("tool_calls", len(session.tool_calls))
        span.set_attribute("citations", len(session.citations))
        _set_output(span, root, "".join(answer_parts))
        while emitted_tool_calls < len(session.tool_calls):
            yield _sse("tool_call", session.tool_calls[emitted_tool_calls])
            emitted_tool_calls += 1
        yield _sse("citations", {"citations": session.citations})
        yield _sse("done", {})
            
def _require_user(authorization: str | None) -> User:
    token = (authorization or "").removeprefix("Bearer ").strip()
    user = get_user(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return user

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    root = trace.get_current_span()
    root.set_attribute("langfuse.trace.name", "NimbusSupport Chat")
    root.set_attribute("langfuse.session.id", request.conversation_id)
    root.set_attribute("langfuse.user.id", user.username)
    root.set_attribute("langfuse.trace.output", request.message)
    return StreamingResponse(_stream(request, user, root), media_type="text/event-stream")
    