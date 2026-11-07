"""Pure helpers for the OpenAI Responses API and token parameters.

This module must not import Home Assistant or openai so it can be unit tested
standalone. Conversation history is kept in Chat Completions message format and
converted to Responses API input at call time.
"""

from __future__ import annotations

from typing import Any

MAX_COMPLETION_TOKENS_MODEL_PREFIXES = ("gpt-4o", "gpt-5", "gpt-6", "o1", "o3", "o4")

REASONING_EFFORT_UNSET_VALUES = (None, "", "default")


def use_max_completion_tokens(model: str | None, force: bool = False) -> bool:
    """Return True if max_completion_tokens should be used instead of max_tokens."""
    if force:
        return True
    model_lower = (model or "").lower()
    return any(
        model_lower.startswith(prefix) or f"-{prefix}" in model_lower
        for prefix in MAX_COMPLETION_TOKENS_MODEL_PREFIXES
    )


def get_token_kwargs(
    model: str | None, max_tokens: int, force: bool = False
) -> dict[str, int]:
    """Return the Chat Completions token limit kwarg for the model."""
    if use_max_completion_tokens(model, force):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def normalize_reasoning_effort(value: str | None) -> str | None:
    """Return the reasoning effort to send, or None when it should be omitted."""
    if value in REASONING_EFFORT_UNSET_VALUES:
        return None
    return value


def to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions tools (or bare function specs) to Responses tools."""
    result = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            spec = tool["function"]
        else:
            spec = tool
        converted = {"type": "function", **spec}
        converted.setdefault("parameters", {"type": "object", "properties": {}})
        # Responses API functions are strict by default, Chat Completions are not.
        converted.setdefault("strict", False)
        result.append(converted)
    return result


def _to_input_content(role: str, content: Any) -> Any:
    """Convert Chat Completions message content to Responses input content."""
    if not isinstance(content, list):
        return "" if content is None else content

    text_type = "output_text" if role == "assistant" else "input_text"
    parts = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": text_type, "text": str(part)})
            continue
        part_type = part.get("type")
        if part_type == "text":
            parts.append({"type": text_type, "text": part.get("text", "")})
        elif part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image = {"type": "input_image", "image_url": image_url.get("url")}
                if image_url.get("detail"):
                    image["detail"] = image_url["detail"]
            else:
                image = {"type": "input_image", "image_url": image_url}
            parts.append(image)
        else:
            parts.append(part)
    return parts


def to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert Chat Completions messages to (instructions, input items)."""
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    pending_legacy_call_ids: list[str] = []

    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role in ("system", "developer"):
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if content:
                instructions.append(content)
        elif role == "user":
            items.append({"role": "user", "content": _to_input_content(role, content)})
        elif role == "assistant":
            if content:
                items.append(
                    {"role": "assistant", "content": _to_input_content(role, content)}
                )
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments") or "{}",
                    }
                )
            function_call = message.get("function_call")
            if function_call:
                # Legacy function calling has no ids, so synthesize one.
                call_id = f"call_legacy_{index}"
                pending_legacy_call_ids.append(call_id)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": function_call.get("name"),
                        "arguments": function_call.get("arguments") or "{}",
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": "" if content is None else str(content),
                }
            )
        elif role == "function":
            if pending_legacy_call_ids:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": pending_legacy_call_ids.pop(0),
                        "output": "" if content is None else str(content),
                    }
                )

    return ("\n\n".join(instructions) or None), items


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_function_calls(response: Any) -> list[dict[str, str]]:
    """Return function_call output items of a Responses API response."""
    return [
        {
            "call_id": _get(item, "call_id"),
            "name": _get(item, "name"),
            "arguments": _get(item, "arguments") or "{}",
        }
        for item in _get(response, "output") or []
        if _get(item, "type") == "function_call"
    ]


def get_output_text(response: Any) -> str:
    """Return the aggregated output text of a Responses API response."""
    text = _get(response, "output_text")
    if text is not None:
        return text
    texts = []
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for part in _get(item, "content") or []:
            if _get(part, "type") == "output_text":
                texts.append(_get(part, "text") or "")
    return "".join(texts)


def is_incomplete_max_tokens(response: Any) -> bool:
    """Return True if the response stopped because of max_output_tokens."""
    if _get(response, "status") != "incomplete":
        return False
    details = _get(response, "incomplete_details")
    return _get(details, "reason") == "max_output_tokens"


def to_assistant_message(
    text: str | None, function_calls: list[dict[str, str]]
) -> dict[str, Any]:
    """Build a Chat Completions assistant message from Responses output."""
    message: dict[str, Any] = {"role": "assistant"}
    if text:
        message["content"] = text
    if function_calls:
        message["tool_calls"] = [
            {
                "id": call["call_id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in function_calls
        ]
    return message
