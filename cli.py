import argparse
import os
import subprocess
import sys
import time
import signal
import json
import urllib.request
from pathlib import Path

from config import MLX_SERVER_BASE_URL, AGENT_DIR, MLX_SERVER_MODEL, MLX_SERVER_PORT

PID_FILE_SERVER = os.path.join(AGENT_DIR, "server.pid")
PID_FILE_AGENT = os.path.join(AGENT_DIR, "agent.pid")
LOG_FILE_SERVER = os.path.join(AGENT_DIR, "server.log")
LOG_FILE_AGENT = os.path.join(AGENT_DIR, "agent.log")

def _write_pid(pid_file, pid):
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)
    with open(pid_file, "w") as f:
        f.write(str(pid))

def _read_pid(pid_file):
    if not os.path.exists(pid_file):
        return None
    with open(pid_file, "r") as f:
        try:
            return int(f.read().strip())
        except ValueError:
            return None

def _remove_pid(pid_file):
    if os.path.exists(pid_file):
        os.remove(pid_file)

def is_running(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def check_server_api():
    try:
        req = urllib.request.Request(f"{MLX_SERVER_BASE_URL}/models")
        with urllib.request.urlopen(req, timeout=1) as response:
            return response.getcode() == 200
    except Exception:
        return False

def start_process(cmd, name, pid_file, log_file, interactive=False):
    if is_running(_read_pid(pid_file)):
        print(f"{name} is already running.")
        return

    print(f"Starting {name}...")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if interactive:
        print(f"{name} runs interactively. We will launch it now. Press Ctrl+C to stop it.")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print(f"\n{name} stopped by user.")
        return

    with open(log_file, "a") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True, # detached
        )
    _write_pid(pid_file, proc.pid)
    print(f"{name} started with PID {proc.pid}. Logs: {log_file}")

def stop_process(name, pid_file):
    pid = _read_pid(pid_file)
    if not is_running(pid):
        print(f"{name} is not running.")
        _remove_pid(pid_file)
        return

    print(f"Stopping {name} (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        if is_running(pid):
            print("Force killing...")
            os.kill(pid, signal.SIGKILL)
    except OSError as e:
        print(f"Error stopping {name}: {e}")
    finally:
        _remove_pid(pid_file)
        print(f"{name} stopped.")

def setup_env():
    print("Setting up agent environment...")
    os.makedirs(AGENT_DIR, exist_ok=True)
    print(f"Created config directory at {AGENT_DIR}")

    print("Checking dependencies...")

    missing_deps = []
    try:
        import mlx_lm
    except Exception as e:
        missing_deps.append(f"mlx_lm ({e})")

    try:
        import sounddevice
    except Exception as e:
        missing_deps.append(f"sounddevice ({e})")

    try:
        import agent_weave
    except Exception as e:
        missing_deps.append(f"agent_weave ({e})")

    if missing_deps:
        print("Warning: Some dependencies could not be loaded:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("Please ensure you run: pip install -r requirements.txt")
        print("Note: 'mlx_lm' requires Apple Silicon (macOS).")
    else:
        print("Dependencies found.")

    print("\nSetup complete. You can now use 'start', 'stop', or 'status' commands.")

def status():
    server_pid = _read_pid(PID_FILE_SERVER)
    agent_pid = _read_pid(PID_FILE_AGENT)

    server_status = "Running" if is_running(server_pid) else "Stopped"
    agent_status = "Running" if is_running(agent_pid) else "Stopped"

    api_status = "Responding" if check_server_api() else "Not responding"

    print("\n=== Agent & Server Status ===")
    print(f"Server: {server_status} (PID: {server_pid or 'N/A'}) - API: {api_status}")
    print(f"Agent:  {agent_status} (PID: {agent_pid or 'N/A'})")
    print("=============================\n")

def main():
    parser = argparse.ArgumentParser(description="Manage the Local Voice Agent and MLX Server.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup
    parser_setup = subparsers.add_parser("setup", help="Initialize configuration and check dependencies")
    parser_setup.add_argument("--force", action="store_true", help="Force setup even if already configured")

    # Start
    parser_start = subparsers.add_parser("start", help="Start the server and agent")
    parser_start.add_argument("--server-only", action="store_true", help="Start only the MLX server")
    parser_start.add_argument("--agent-only", action="store_true", help="Start only the agent")
    parser_start.add_argument("--voice", action="store_true", help="Start the voice agent (run.py) instead of text chat")
    parser_start.add_argument("--detached", action="store_true", help="Run agent detached (default is interactive for agent)")

    # Stop
    parser_stop = subparsers.add_parser("stop", help="Stop the server and agent")
    parser_stop.add_argument("--server-only", action="store_true", help="Stop only the MLX server")
    parser_stop.add_argument("--agent-only", action="store_true", help="Stop only the agent")

    # Status
    parser_status = subparsers.add_parser("status", help="Show the status of server and agent")

    args = parser.parse_args()

    if args.command == "setup":
        setup_env()

    elif args.command == "start":
        if not args.agent_only:
            start_process(
                [sys.executable, "-m", "mlx_lm.server", "--model", MLX_SERVER_MODEL, "--port", str(MLX_SERVER_PORT)],
                "MLX Server",
                PID_FILE_SERVER,
                LOG_FILE_SERVER,
                interactive=False # Server is always detached
            )
            if not args.server_only:
                print("Waiting for server to be ready...")
                for _ in range(30):
                    if check_server_api():
                        print("Server is ready.")
                        break
                    time.sleep(1)
                else:
                    print("Warning: Server may not be ready yet.")

        if not args.server_only:
            script = "run.py" if args.voice else "chat.py"
            # Agents require stdin, so by default interactive=True unless specified
            interactive = not args.detached
            start_process(
                [sys.executable, script],
                "Agent Interface",
                PID_FILE_AGENT,
                LOG_FILE_AGENT,
                interactive=interactive
            )

    elif args.command == "stop":
        if not args.agent_only:
            stop_process("MLX Server", PID_FILE_SERVER)
        if not args.server_only:
            stop_process("Agent Interface", PID_FILE_AGENT)

    elif args.command == "status":
        status()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
