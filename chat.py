#!/usr/bin/env python3
"""Text-only chat with the full-capability local agent."""

import sys
from rich.console import Console

from config import MLX_SERVER_BASE_URL, MLX_SERVER_MODEL
from agent_weave import Agent
from agent_weave.llm.openai_backend import OpenAIBackend
from agent_weave.memory import ConversationMemory
from tools_full import ALL_TOOLS

console = Console()

SYSTEM_PROMPT = (
    "You are a powerful local AI agent running on the user's Mac. "
    "You have a comprehensive toolset covering:\n\n"
    "- File operations: read, write, edit, create folders, list, copy, move\n"
    "- Shell and Python execution\n"
    "- Web search, webpage fetching, arbitrary HTTP/API requests\n"
    "- MCP server integration (list and call remote tools)\n"
    "- Persistent memory (save, read, search)\n"
    "- Self-learning skills (create, view, list, delete)\n"
    "- Scheduled tasks\n"
    "- Email (send via SMTP, read via IMAP)\n"
    "- Math and utilities\n\n"
    "Guidelines:\n"
    "- Use tools proactively. If the user asks you to do something, do it.\n"
    "- For multi-step tasks, chain tool calls until complete.\n"
    "- Before destructive operations (deleting, overwriting), confirm with the user.\n"
    "- When you figure out a reusable workflow, save it with skill_create.\n"
    "- Save important user facts and preferences with save_memory.\n"
    "- Read memory at the start of a session to recall context.\n"
    "- Keep responses concise.\n"
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
    max_iterations=15,
    verbose=False,
)

console.print(f"[bold cyan]Full agent ready — {len(ALL_TOOLS)} tools loaded.[/bold cyan]")
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