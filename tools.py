"""Tool definitions for the local voice agent.

The agent_weave @tool decorator auto-generates the JSON schema from
the function's type hints, so keep hints accurate.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

from agent_weave import tool

# ── Sandbox: only these directories are accessible ─────────────────────
ALLOWED_ROOTS = [
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "Projects",
    Path.home() / ".my_voice_agent",
    Path.cwd(),
]

BLOCKED_COMMANDS = [
    "rm -rf /",
    "sudo rm",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
    "> /dev/sda",
    "shutdown",
    "reboot",
]


def _is_path_allowed(path: str) -> bool:
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    for root in ALLOWED_ROOTS:
        try:
            r = root.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved == r or r in resolved.parents:
            return True
    return False


@tool(description=(
    "Run a shell command and return its stdout and stderr. "
    "Use for file listing, grep, git, and similar tasks. "
    "Destructive commands are blocked."
))
def execute_shell(command: str) -> str:
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            return f"[BLOCKED] Command contains prohibited pattern: '{blocked}'"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path.cwd()),
        )
        out = result.stdout
        if result.stderr:
            out += f"\n[stderr]\n{result.stderr}"
        if not out.strip():
            out = f"[exit code {result.returncode}]"
        return out[:8000]
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 30 seconds."
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description=(
    "Read a text file and return its contents. "
    "Only files inside the sandboxed directories can be read."
))
def read_file(path: str) -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Access to '{path}' is not permitted."
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return f"[ERROR] Not a file: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:16000]
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description=(
    "Write text content to a file, creating parent directories as needed. "
    "Only paths inside the sandboxed directories are writable."
))
def write_file(path: str, content: str) -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Writing to '{path}' is not permitted."
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[OK] Wrote {len(content)} characters to {p}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description=(
    "List the files and directories at a given path. "
    "Only paths inside the sandboxed directories can be listed."
))
def list_directory(path: str = ".") -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Access to '{path}' is not permitted."
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return f"[ERROR] Not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries[:200]:
            icon = "DIR " if entry.is_dir() else "FILE"
            size = ""
            if entry.is_file():
                size = f"  ({entry.stat().st_size:,} bytes)"
            lines.append(f"{icon} {entry.name}{size}")
        return "\n".join(lines) if lines else "[empty directory]"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description=(
    "Search the web for a query and return the top results. "
    "Useful for current information, documentation, and news."
))
def web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}\n")
        return "\n---\n".join(results) if results else "[no results found]"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description=(
    "Fetch a web page by URL and return its visible text, with HTML, "
    "scripts, and navigation stripped out."
))
def fetch_webpage(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:12000]
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Get the current local date and time on this system.")
def get_current_time() -> str:
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


@tool(description=(
    "Save a note to the agent's persistent memory. Use this to remember "
    "important facts, user preferences, or context across conversations."
))
def save_memory(note: str) -> str:
    from config import MEMORY_DIR
    memory_file = Path(MEMORY_DIR) / "MEMORY.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(memory_file, "a", encoding="utf-8") as f:
        f.write(f"\n- [{ts}] {note}\n")
    return f"[OK] Saved to memory: {note}"


@tool(description=(
    "Read all saved memory notes. Call this at the start of a session "
    "to recall previously saved facts and user preferences."
))
def read_memory() -> str:
    from config import MEMORY_DIR
    memory_file = Path(MEMORY_DIR) / "MEMORY.md"
    if not memory_file.exists():
        return "[no memories saved yet]"
    return memory_file.read_text(encoding="utf-8")[:8000]


ALL_TOOLS = [
    execute_shell,
    read_file,
    write_file,
    list_directory,
    web_search,
    fetch_webpage,
    get_current_time,
    save_memory,
    read_memory,
]
