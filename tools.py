"""Full-capability tool suite for the local agent.

Includes file ops, shell, web, HTTP API, MCP client, memory,
scheduling, skills, email, and utility tools.
"""

import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import requests

from agent_weave import tool

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

from config import MEMORY_DIR

MEMORY_PATH = Path(MEMORY_DIR)

ALLOWED_ROOTS = [
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "Projects",
    Path.home() / ".my_voice_agent",
    Path.home() / "local-voice-agent",
    Path.cwd(),
]

BLOCKED_COMMANDS = [
    "rm -rf /", "sudo rm", "mkfs", "dd if=",
    ":(){ :|:& };:", "> /dev/sda", "shutdown", "reboot",
]

WORKSPACE = Path.home() / ".my_voice_agent"
WORKSPACE.mkdir(parents=True, exist_ok=True)
SKILLS_DIR = WORKSPACE / "skills"
SKILLS_DIR.mkdir(exist_ok=True)
CRON_DIR = WORKSPACE / "cron"
CRON_DIR.mkdir(exist_ok=True)


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


# ═══════════════════════════════════════════════════════════════════════
#  FILE SYSTEM TOOLS
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Read a text file and return its contents. Sandboxed to allowed directories.")
def read_file(path: str) -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Access to '{path}' not permitted."
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return f"[ERROR] Not a file: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:16000]
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Write text content to a file, creating parent dirs. Sandboxed.")
def write_file(path: str, content: str) -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Writing to '{path}' not permitted."
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[OK] Wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Edit a file by replacing old_text with new_text. Use for surgical edits instead of rewriting the whole file.")
def edit_file(path: str, old_text: str, new_text: str) -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Access to '{path}' not permitted."
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return f"[ERROR] Not a file: {path}"
        content = p.read_text(encoding="utf-8")
        if old_text not in content:
            return f"[ERROR] old_text not found in {path}"
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"[OK] Edited {p} (replaced {len(old_text)} chars)"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Create a folder (and any parents). Sandboxed.")
def create_folder(path: str) -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Creating '{path}' not permitted."
    try:
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return f"[OK] Created folder {p}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="List files and folders at a path. Sandboxed.")
def list_directory(path: str = ".") -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Access to '{path}' not permitted."
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return f"[ERROR] Not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries[:200]:
            icon = "DIR " if entry.is_dir() else "FILE"
            size = f"  ({entry.stat().st_size:,} bytes)" if entry.is_file() else ""
            lines.append(f"{icon} {entry.name}{size}")
        return "\n".join(lines) if lines else "[empty]"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Copy a file from src to dst. Sandboxed.")
def copy_file(src: str, dst: str) -> str:
    if not _is_path_allowed(src) or not _is_path_allowed(dst):
        return "[BLOCKED] Source or destination not permitted."
    try:
        import shutil
        s = Path(src).expanduser().resolve()
        d = Path(dst).expanduser().resolve()
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        return f"[OK] Copied {s} -> {d}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Move/rename a file or folder. Sandboxed.")
def move_file(src: str, dst: str) -> str:
    if not _is_path_allowed(src) or not _is_path_allowed(dst):
        return "[BLOCKED] Source or destination not permitted."
    try:
        import shutil
        s = Path(src).expanduser().resolve()
        d = Path(dst).expanduser().resolve()
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return f"[OK] Moved {s} -> {d}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Get metadata about a file: size, modified time, type.")
def file_info(path: str) -> str:
    if not _is_path_allowed(path):
        return f"[BLOCKED] Access to '{path}' not permitted."
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"[ERROR] Path does not exist: {path}"
        st = p.stat()
        return json.dumps({
            "path": str(p),
            "type": "directory" if p.is_dir() else "file",
            "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(st.st_ctime).isoformat(),
        }, indent=2)
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  SHELL / CODE EXECUTION
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Run a shell command and return stdout/stderr. Destructive commands blocked.")
def execute_shell(command: str, cwd: str = "") -> str:
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            return f"[BLOCKED] Prohibited pattern: '{blocked}'"
    try:
        workdir = str(Path(cwd).expanduser().resolve()) if cwd else str(Path.cwd())
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=workdir,
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


@tool(description="Execute Python code in a subprocess and return the result. Use for calculations, data processing, and scripting.")
def execute_python(code: str) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        out = result.stdout
        if result.stderr:
            out += f"\n[stderr]\n{result.stderr}"
        return out[:8000] if out.strip() else f"[exit code {result.returncode}]"
    except subprocess.TimeoutExpired:
        return "[ERROR] Python execution timed out (30s)."
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  WEB / HTTP / API
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Search the web via DuckDuckGo and return top results.")
def web_search(query: str, max_results: int = 5) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                href = r.get("href") or r.get("url", "")
                body = r.get("body") or r.get("description", "")
                results.append(f"**{title}**\n{href}\n{body}\n")
        return "\n---\n".join(results) if results else "[no results found]"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Fetch a web page and return its visible text.")
def fetch_webpage(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:12000]
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Make an arbitrary HTTP request to an API. Supports GET, POST, PUT, DELETE, PATCH. Returns the response body (truncated) and status code.")
def http_request(url: str, method: str = "GET", headers_json: str = "{}",
                 body_json: str = "", timeout: int = 30) -> str:
    """Make an HTTP API request. headers_json and body_json are JSON strings."""
    try:
        headers = json.loads(headers_json) if headers_json else {}
        body = json.loads(body_json) if body_json else None
        method = method.upper()
        resp = requests.request(
            method, url, headers=headers, json=body, timeout=timeout,
        )
        text = resp.text[:8000]
        return f"[HTTP {resp.status_code}]\n{text}"
    except json.JSONDecodeError as e:
        return f"[ERROR] Invalid JSON in headers or body: {e}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  MCP CLIENT
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Connect to an MCP server over HTTP/SSE and list its available tools. The server URL should be the MCP endpoint. Supports the Streamable HTTP transport.")
def mcp_list_tools(server_url: str) -> str:
    """List tools from an MCP server. Returns JSON of tool definitions."""
    try:
        resp = requests.post(
            server_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=15,
        )
        if resp.status_code != 200:
            return f"[MCP ERROR] HTTP {resp.status_code}: {resp.text[:500]}"
        # Handle SSE-style response
        text = resp.text
        if text.startswith("event:") or "data:" in text:
            for line in text.splitlines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    return json.dumps(data.get("result", data), indent=2)[:8000]
        return json.dumps(resp.json(), indent=2)[:8000]
    except Exception as e:
        return f"[MCP ERROR] {type(e).__name__}: {e}"


@tool(description="Call a tool on a remote MCP server. Pass arguments as a JSON string.")
def mcp_call_tool(server_url: str, tool_name: str, arguments_json: str = "{}") -> str:
    """Call a tool on an MCP server. arguments_json is a JSON object string."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
        resp = requests.post(
            server_url,
            json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=60,
        )
        if resp.status_code != 200:
            return f"[MCP ERROR] HTTP {resp.status_code}: {resp.text[:500]}"
        text = resp.text
        if "data:" in text:
            for line in text.splitlines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    return json.dumps(data.get("result", data), indent=2)[:8000]
        return json.dumps(resp.json(), indent=2)[:8000]
    except Exception as e:
        return f"[MCP ERROR] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  MEMORY
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Save a note to persistent memory. Use for facts, preferences, and context across sessions.")
def save_memory(note: str, category: str = "general") -> str:
    try:
        memory_file = MEMORY_PATH / "MEMORY.md"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(f"\n- [{ts}] ({category}) {note}\n")
        return f"[OK] Saved to memory ({category}): {note}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Read all saved memory notes.")
def read_memory() -> str:
    try:
        memory_file = MEMORY_PATH / "MEMORY.md"
        if not memory_file.exists():
            return "[no memories saved yet]"
        return memory_file.read_text(encoding="utf-8")[:8000]
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Search saved memory notes for a keyword or phrase.")
def search_memory(query: str) -> str:
    try:
        memory_file = MEMORY_PATH / "MEMORY.md"
        if not memory_file.exists():
            return "[no memories saved yet]"
        lines = memory_file.read_text(encoding="utf-8").splitlines()
        query_lower = query.lower()
        matches = [l for l in lines if query_lower in l.lower()]
        return "\n".join(matches) if matches else f"[no memories matching '{query}']"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="List all local tools available in this agent (names only). Use this when the user asks 'list tools', 'what can you do', or 'help'. Do NOT call mcp_list_tools for that.")
def list_local_tools() -> str:
    return "\n".join(sorted([
        "read_file", "write_file", "edit_file", "create_folder",
        "list_directory", "copy_file", "move_file", "file_info",
        "execute_shell", "execute_python",
        "web_search", "fetch_webpage", "http_request",
        "mcp_list_tools", "mcp_call_tool",
        "save_memory", "read_memory", "search_memory",
        "skill_create", "skill_view", "skills_list", "skill_delete",
        "cronjob_manage",
        "get_current_time", "generate_uuid", "calculate",
        "send_email", "check_email", "list_local_tools",
    ]))


# ═══════════════════════════════════════════════════════════════════════
#  SKILLS (self-learning, Hermes-style)
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Save a reusable skill/procedure to disk. Skills are markdown documents the agent can load later. Use this when you figure out a non-trivial workflow worth remembering.")
def skill_create(name: str, content: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())
    path = SKILLS_DIR / f"{slug}.md"
    path.write_text(f"# {name}\n\n{content}", encoding="utf-8")
    return f"[OK] Saved skill '{name}' to {path}"


@tool(description="View a saved skill by name or slug.")
def skill_view(name: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())
    path = SKILLS_DIR / f"{slug}.md"
    if not path.exists():
        return f"[ERROR] Skill '{name}' not found."
    return path.read_text(encoding="utf-8")[:8000]


@tool(description="List all saved skills.")
def skills_list() -> str:
    files = sorted(SKILLS_DIR.glob("*.md"))
    if not files:
        return "[no skills saved yet]"
    return "\n".join(f"- {f.stem}" for f in files)


@tool(description="Delete a saved skill by name.")
def skill_delete(name: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())
    path = SKILLS_DIR / f"{slug}.md"
    if not path.exists():
        return f"[ERROR] Skill '{name}' not found."
    path.unlink()
    return f"[OK] Deleted skill '{name}'"


# ═══════════════════════════════════════════════════════════════════════
#  SCHEDULING / CRON
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Create a scheduled task. The task is a shell command that runs at the given interval. Interval accepts values like '1h', '30m', '1d', or a standard cron expression.")
def cronjob_manage(action: str, name: str = "", command: str = "",
                   interval: str = "") -> str:
    """Manage scheduled tasks. action is 'create', 'list', or 'delete'."""
    task_file = CRON_DIR / "tasks.json"
    tasks = {}
    if task_file.exists():
        try:
            tasks = json.loads(task_file.read_text())
        except Exception:
            tasks = {}

    if action == "list":
        if not tasks:
            return "[no scheduled tasks]"
        return json.dumps(tasks, indent=2)

    if action == "create":
        if not name or not command:
            return "[ERROR] name and command are required for create"
        tasks[name] = {
            "command": command,
            "interval": interval or "1h",
            "created": datetime.now().isoformat(),
        }
        task_file.write_text(json.dumps(tasks, indent=2))
        return f"[OK] Created scheduled task '{name}'"

    if action == "delete":
        if name in tasks:
            del tasks[name]
            task_file.write_text(json.dumps(tasks, indent=2))
            return f"[OK] Deleted task '{name}'"
        return f"[ERROR] Task '{name}' not found"

    return f"[ERROR] Unknown action: {action}"


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Get the current local date and time.")
def get_current_time() -> str:
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


@tool(description="Generate a random UUID (v4).")
def generate_uuid() -> str:
    return str(uuid.uuid4())


@tool(description="Evaluate a mathematical expression safely. Supports + - * / ** % and parentheses.")
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "[ERROR] Expression contains disallowed characters."
    try:
        # safe math eval
        import ast
        import operator

        def safe_eval(expr):
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.Pow: operator.pow, ast.BitXor: operator.xor,
                   ast.USub: operator.neg, ast.Mod: operator.mod}

            def eval_node(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_node(node.operand))
                elif isinstance(node, ast.Tuple):
                    return tuple(eval_node(n) for n in node.elts)
                else:
                    raise TypeError(node)
            parsed = ast.parse(expr.strip(), mode='eval').body
            return eval_node(parsed)

        result = safe_eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  EMAIL (SMTP send + IMAP read)
# ═══════════════════════════════════════════════════════════════════════

@tool(description="Send an email via SMTP. Requires SMTP_HOST, SMTP_USER, SMTP_PASS in environment variables.")
def send_email(to: str, subject: str, body: str) -> str:
    import smtplib
    from email.mime.text import MIMEText
    host = os.environ.get("SMTP_HOST", "")
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")
    if not host or not user:
        return "[ERROR] SMTP_HOST, SMTP_USER, SMTP_PASS not set in environment."
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        with smtplib.SMTP(host, 587) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        return f"[OK] Sent email to {to}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@tool(description="Read recent emails via IMAP. Requires IMAP_HOST, IMAP_USER, IMAP_PASS in environment variables.")
def check_email(limit: int = 10) -> str:
    import imaplib
    import email as email_lib
    host = os.environ.get("IMAP_HOST", "")
    user = os.environ.get("IMAP_USER", "")
    password = os.environ.get("IMAP_PASS", "")
    if not host or not user:
        return "[ERROR] IMAP_HOST, IMAP_USER, IMAP_PASS not set in environment."
    try:
        with imaplib.IMAP4_SSL(host) as m:
            m.login(user, password)
            m.select("INBOX")
            _, data = m.search(None, "ALL")
            ids = data[0].split()[-limit:]
            results = []
            if ids:
                fetch_ids = b",".join(ids)
                _, msg_data = m.fetch(fetch_ids, "(RFC822)")
                # msg_data contains tuples of (header, message) and then a b')' for each message.
                for part in msg_data:
                    if isinstance(part, tuple):
                        msg = email_lib.message_from_bytes(part[1])
                        results.append(
                            f"From: {msg['From']}\nSubject: {msg['Subject']}\n"
                            f"Date: {msg['Date']}\n"
                        )
                results.reverse()
            return "\n---\n".join(results) if results else "[no emails]"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    # File system
    read_file, write_file, edit_file, create_folder, list_directory,
    copy_file, move_file, file_info,
    # Shell / code
    execute_shell, execute_python,
    # Web / HTTP
    web_search, fetch_webpage, http_request,
    # MCP
    mcp_list_tools, mcp_call_tool,
    # Memory
    save_memory, read_memory, search_memory,
    # Skills
    skill_create, skill_view, skills_list, skill_delete,
    # Scheduling
    cronjob_manage,
    # Utilities
    get_current_time, generate_uuid, calculate,
    # Email
    send_email, check_email,
    # Introspection (answer "list tools" locally, never via MCP)
    list_local_tools,
]
