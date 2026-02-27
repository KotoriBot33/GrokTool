"""
Responses API bridge service (OpenAI-compatible, lightweight).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import orjson

from app.services.grok.chat import ChatService


def _now_ts() -> int:
    return int(time.time())


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _normalize_reasoning_effort(reasoning: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort") or reasoning.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        return effort.strip()
    return None


def _coerce_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text"}:
                blocks.append({"type": "text", "text": item.get("text", "")})
                continue
            if item_type in {"input_image", "image", "image_url", "output_image"}:
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url") or ""
                    detail = image_url.get("detail")
                    payload = {"url": url} if url else None
                    if payload and detail:
                        payload["detail"] = detail
                    if payload:
                        blocks.append({"type": "image_url", "image_url": payload})
                elif isinstance(image_url, str) and image_url:
                    blocks.append({"type": "image_url", "image_url": {"url": image_url}})
                elif isinstance(item.get("url"), str) and item.get("url"):
                    blocks.append({"type": "image_url", "image_url": {"url": item.get("url")}})
                continue
            if item_type in {"input_file", "file"}:
                file_data = item.get("file_data")
                file_id = item.get("file_id")
                if isinstance(item.get("file"), dict):
                    file_data = file_data or item["file"].get("file_data")
                    file_id = file_id or item["file"].get("file_id")
                file_payload: Dict[str, Any] = {}
                if file_data:
                    file_payload["file_data"] = file_data
                if file_id:
                    file_payload["file_id"] = file_id
                if file_payload:
                    blocks.append({"type": "file", "file": file_payload})
                continue
            if item_type in {"input_audio", "audio"}:
                audio = item.get("audio") if isinstance(item.get("audio"), dict) else {}
                data = audio.get("data") or item.get("data")
                if data:
                    blocks.append({"type": "input_audio", "input_audio": {"data": data}})
                continue
        return blocks if blocks else ""
    return str(content)


def _coerce_input_to_messages(input_value: Any) -> List[Dict[str, Any]]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, dict):
        if "role" in input_value and "content" in input_value:
            return [{"role": input_value.get("role") or "user", "content": _coerce_content(input_value.get("content"))}]
        content = _coerce_content(input_value)
        if content:
            return [{"role": "user", "content": content}]
        return []

    if not isinstance(input_value, list):
        return [{"role": "user", "content": str(input_value)}]

    messages: List[Dict[str, Any]] = []
    pending_blocks: List[Dict[str, Any]] = []

    def _flush_pending() -> None:
        nonlocal pending_blocks
        if pending_blocks:
            messages.append({"role": "user", "content": pending_blocks})
            pending_blocks = []

    for item in input_value:
        if isinstance(item, str):
            pending_blocks.append({"type": "text", "text": item})
            continue

        if isinstance(item, dict):
            if "role" in item and "content" in item:
                _flush_pending()
                messages.append({"role": item.get("role") or "user", "content": _coerce_content(item.get("content"))})
                continue

            item_type = item.get("type")
            if item_type in {"tool_output", "function_call_output", "tool_call_output", "input_tool_output"}:
                _flush_pending()
                call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id") or _new_tool_call_id()
                output = item.get("output") or item.get("content") or ""
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
                continue

            content = _coerce_content(item)
            if isinstance(content, list):
                pending_blocks.extend(content)
            elif isinstance(content, str) and content:
                pending_blocks.append({"type": "text", "text": content})
            continue

        pending_blocks.append({"type": "text", "text": str(item)})

    _flush_pending()
    return messages


def _messages_from_input(input_value: Any, instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    messages = _coerce_input_to_messages(input_value)
    if instructions and instructions.strip():
        messages.insert(0, {"role": "system", "content": instructions.strip()})
    if not messages:
        messages = [{"role": "user", "content": ""}]
    return messages


def _usage_from_chat(chat_result: Dict[str, Any]) -> Dict[str, int]:
    usage = chat_result.get("usage") if isinstance(chat_result, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _build_response_json(
    *,
    model: str,
    text: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, int]] = None,
    response_id: Optional[str] = None,
) -> Dict[str, Any]:
    rid = response_id or _new_response_id()
    output_blocks: List[Dict[str, Any]] = []

    if text is not None:
        output_blocks.append({
            "id": _new_message_id(),
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": text, "annotations": []}
            ],
        })

    if tool_calls:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = func.get("name") or "function"
            arguments = func.get("arguments") or "{}"
            output_blocks.append({
                "id": tc.get("id") or _new_function_call_id(),
                "type": "function_call",
                "call_id": tc.get("id") or _new_tool_call_id(),
                "name": name,
                "arguments": arguments,
            })

    return {
        "id": rid,
        "object": "response",
        "created_at": _now_ts(),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output_blocks,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "user": None,
        "metadata": {},
    }


def _chat_to_response_json(model: str, chat_result: Dict[str, Any], response_id: Optional[str] = None) -> Dict[str, Any]:
    choices = chat_result.get("choices") if isinstance(chat_result, dict) else None
    message = None
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") if isinstance(first.get("message"), dict) else None

    text: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif content is None:
            text = None
        else:
            text = str(content)

        tc = message.get("tool_calls")
        if isinstance(tc, list):
            tool_calls = tc

    usage = _usage_from_chat(chat_result)
    return _build_response_json(model=model, text=text, tool_calls=tool_calls, usage=usage, response_id=response_id)


async def _responses_stream_from_chat(
    chat_stream: AsyncGenerator[str, None],
    model: str,
) -> AsyncGenerator[str, None]:
    response_id = _new_response_id()
    message_id = _new_message_id()
    created_at = _now_ts()
    text_acc: List[str] = []
    tool_calls_buffer: List[Dict[str, Any]] = []

    created_event = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": model,
            "output": [],
        },
    }
    yield f"data: {orjson.dumps(created_event).decode()}\n\n"

    try:
        async for raw_chunk in chat_stream:
            if not isinstance(raw_chunk, str):
                continue

            lines = raw_chunk.splitlines()
            for line in lines:
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    item = orjson.loads(payload)
                except Exception:
                    continue

                choices = item.get("choices") if isinstance(item, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

                part = delta.get("content")
                if isinstance(part, str) and part:
                    text_acc.append(part)
                    evt = {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": part,
                    }
                    yield f"data: {orjson.dumps(evt).decode()}\n\n"

                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    tool_calls_buffer.extend([tc for tc in tool_calls if isinstance(tc, dict)])

        full_text = "".join(text_acc)
        done_evt = {
            "type": "response.output_text.done",
            "response_id": response_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        }
        yield f"data: {orjson.dumps(done_evt).decode()}\n\n"

        completed_response = _build_response_json(
            model=model,
            text=full_text,
            tool_calls=tool_calls_buffer or None,
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            response_id=response_id,
        )
        completed_response["created_at"] = created_at
        completed_response["status"] = "completed"

        completed_evt = {"type": "response.completed", "response": completed_response}
        yield f"data: {orjson.dumps(completed_evt).decode()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_evt = {
            "type": "response.error",
            "response_id": response_id,
            "error": {"message": str(e), "type": "server_error"},
        }
        yield f"data: {orjson.dumps(error_evt).decode()}\n\n"
        yield "data: [DONE]\n\n"


class ResponsesService:
    """Bridge OpenAI Responses API to existing ChatService."""

    @staticmethod
    async def create(
        *,
        model: str,
        input_value: Any,
        instructions: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = True,
        reasoning: Optional[Dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        store: Optional[bool] = None,
        previous_response_id: Optional[str] = None,
        truncation: Optional[str] = None,
    ):
        del max_output_tokens, metadata, user, store, previous_response_id, truncation  # reserved

        messages = _messages_from_input(input_value, instructions)
        reasoning_effort = _normalize_reasoning_effort(reasoning)

        # map to legacy thinking switch for compatibility
        thinking = None
        if isinstance(reasoning_effort, str):
            thinking = "disabled" if reasoning_effort == "none" else "enabled"

        result = await ChatService.completions(
            model=model,
            messages=messages,
            stream=stream,
            thinking=thinking,
        )

        if stream:
            return _responses_stream_from_chat(result, model)

        if not isinstance(result, dict):
            # fallback: if upstream unexpectedly streamed in non-stream mode
            return _build_response_json(model=model, text="", usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        return _chat_to_response_json(model, result)
