import os
import json
import asyncio
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv

# ---- Load .env ----
load_dotenv()

# ---- Configure Gemini ----
API_KEY = os.getenv("GOOGLE_API_KEY1")
if not API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your .env file")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

MAX_MESSAGES = 10
CONVERSATION_LOG = "conversation_log.json"

# ---- Seed with system prompt ----
from .prompts import system_prompt
messages = [{"role": "model", "parts": [{"text": system_prompt}]}]


# ---------------- Helpers ----------------
def log_message(level: str, message: str):
    logos = {
        "info": "🟢 [BOT INFO]",
        "warn": "🟡 [BOT WARN]",
        "error": "🔴 [BOT ERROR]",
        "ask": "❓ [BOT ASK]",
        "brain": "🧠 [BOT PLAN]",
        "bot": "🤖 [BOT]",
    }
    prefix = logos.get(level.lower(), "ℹ️ [BOT]")
    print(f"{prefix} {message}")


def trim_history(msgs, max_len=MAX_MESSAGES):
    if len(msgs) <= max_len:
        return msgs
    return [msgs[0]] + msgs[-(max_len - 1):]


def append_to_conversation_log(user_query, assistant_response):
    log_entry = {"user_query": user_query, "assistant_response": assistant_response}
    try:
        try:
            with open(CONVERSATION_LOG, "r") as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []

        log_data.append(log_entry)
        with open(CONVERSATION_LOG, "w") as f:
            json.dump(log_data, f, indent=2)
        log_message("info", f"Logged conversation to {CONVERSATION_LOG}")
    except Exception as e:
        log_message("warn", f"Failed to log conversation: {e}")


def extract_all_json(text: str) -> List[Dict[str, Any]]:
    """Extract all balanced JSON objects from text."""
    results = []
    s = text
    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = s[start:i+1]
                try:
                    results.append(json.loads(candidate))
                except Exception:
                    pass
                start = None
    return results


# ---------------- Core Function ----------------
def process_user_query(user_query: str) -> Dict[str, Any]:
    """
    Process a user query through Gemini and return structured results.

    Returns dict with:
        - raw: raw Gemini text
        - steps: list of extracted JSON steps
        - final: final user-facing text
    """
    global messages
    chat = model.start_chat(history=messages)

    # Add user query
    messages.append({"role": "user", "parts": [{"text": user_query}]})
    messages = trim_history(messages)

    try:
        resp = chat.send_message(user_query)
    except Exception as e:
        return {"raw": "", "steps": [], "final": f"❌ Gemini API error: {e}"}

    raw = resp.text.strip()
    steps = extract_all_json(raw)

    # Default fallback
    final_output = None
    for step in steps:
        stype = (step.get("step") or "").lower()
        if stype == "output":
            final_output = step.get("content") or step.get("output")

    reply = final_output if final_output else raw

    # Save
    messages.append({"role": "model", "parts": [{"text": reply}]})
    messages = trim_history(messages)
    append_to_conversation_log(user_query, reply)

    return {
        "final": reply  # Final answer for user
    }



