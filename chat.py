#!/usr/bin/env python3
"""Text-only chat with the full-capability local agent."""

# Disable Qwen3 thinking mode globally so tool calls don't time out.
try:
    import openai.resources.chat.completions as _cc
    _orig_create = _cc.Completions.create

    def _patched_create(self, *args, **kwargs):
        extra = kwargs.setdefault("extra_body", {})
        ctk = extra.setdefault("chat_template_kwargs", {})
        ctk["enable_thinking"] = False
        return _orig_create(self, *args, **kwargs)

    _cc.Completions.create = _patched_create
except Exception as _e:
    print(f"[warn] Could not patch openai for thinking mode: {_e}")

import sys
import os
import platform
from pathlib import Path
from rich.console import Console

from config import MLX_SERVER_BASE_URL, MLX_SERVER_MODEL
from agent_weave import Agent, AgentSettings
from agent_weave.llm.openai_backend import OpenAIBackend
from agent_weave.memory import ConversationMemory
from tools_full import ALL_TOOLS

console = Console()

SYSTEM_PROMPT = (
    "You are a powerful local AI agent running on the user's Mac with full "
    "file-system, shell, web, and API access via tools.\n\n"
    "CRITICAL RULES - follow these exactly:\n\n"
    "1. ALWAYS CALL THE TOOL. Never say 'I cannot' or 'I don't have access' "
    "without first calling the relevant tool. If unsure whether you can do "
    "something, CALL THE TOOL and let it return the real result. The tools "
    "return [ERROR] messages when something actually fails - trust those, "
    "not your assumptions.\n\n"
    "2. PRESERVE USER INTENT. If the user asks for a haiku about lobsters, "
    "the output must be about lobsters. If they specify a path, use that "
    "exact path. Do not substitute your own subject or location.\n\n"
    "3. PATHS: Tilde (~) expands to the home directory. Both "
    "~/Documents/agent-test/foo.txt and /Users/ttk/Documents/agent-test/foo.txt "
    "are valid. Use the path the user gave you.\n\n"
    "4. PATH CORRECTION: If the user's path may not be exact, and a path in "
    "close spelling proximity exists, ask the user: 'Did you mean "
    "<suggested_path>?' and then use that path after confirmation. Do not "
    "silently substitute a different path.\n\n"
    "5. CHAIN TOOL CALLS. For multi-step tasks (search then save, read then "
    "edit), call each tool in sequence. Do not stop after one call if the "
    "task requires more.\n\n"
    "6. REPORT FAITHFULLY. When a tool returns [ERROR], say exactly what the "
    "error was. Do not fabricate a success or invent a plausible-sounding "
    "result.\n\n"
    "7. BE CONCISE. Keep replies short. Show the result, not a lecture.\n\n"
    "Available tools: read_file, write_file, edit_file, create_folder, "
    "list_directory, copy_file, move_file, file_info, execute_shell, "
    "execute_python, web_search, fetch_webpage, http_request, "
    "mcp_list_tools, mcp_call_tool, save_memory, read_memory, search_memory, "
    "skill_create, skill_view, skills_list, skill_delete, cronjob_manage, "
    "get_current_time, generate_uuid, calculate, send_email, check_email, "
    "list_local_tools.\n\n"
    f"Environment: OS={platform.system()} {platform.release()}, "
    f"home={Path.home()}, cwd={os.getcwd()}.\n"
    "Rules: resolve relative paths against cwd, never invent paths "
    "(never use /Users/tk, use the real home above). "
    "When the user says 'list tools' or 'what can you do', call "
    "list_local_tools, never mcp_list_tools. "
    "If a tool returns [ERROR]/[BLOCKED], report it verbatim and do not "
    "retry the same call more than twice."
)

backend = OpenAIBackend(
    api_key="local",
    base_url=MLX_SERVER_BASE_URL,
    default_model=MLX_SERVER_MODEL,
)

agent = Agent(
    name="local-full-agent",
    llm=backend,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
    memory=ConversationMemory(),
    model=MLX_SERVER_MODEL,
    settings=AgentSettings(
        default_model=MLX_SERVER_MODEL,
        max_iterations=15,
        request_timeout=300.0,
        verbose=False,
    ),
    verbose=False,
)

console.print(f"[bold cyan]Full agent ready - {len(ALL_TOOLS)} tools loaded.[/bold cyan]")
console.print(f"[dim]Model: {MLX_SERVER_MODEL}[/dim]")
console.print("[dim]Type 'quit' to exit.[/dim]\n")

while True:
    try:
        text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not text:
        continue
    if text.lower() in ("quit", "exit", "q"):
        break

    try:
        result = agent.run(text)
        console.print(f"[bold green]Agent:[/bold green] {result.output}\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {type(e).__name__}: {e}\n")
