#!/usr/bin/env python3
"""Main entry point for the local voice agent.

Usage:
    Terminal 1:  python run.py --server
    Terminal 2:  python run.py

Or single command:
    python run.py --auto-server
"""

import argparse
import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel

from config import (
    NOISE_MULTIPLIER,
    MIN_NOISE_FLOOR,
    MLX_SERVER_HOST,
    MLX_SERVER_PORT,
    MLX_SERVER_MODEL,
    MLX_SERVER_BASE_URL,
)

console = Console()


def start_mlx_server() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.server",
        "--model",
        MLX_SERVER_MODEL,
        "--host",
        MLX_SERVER_HOST,
        "--port",
        str(MLX_SERVER_PORT),
    ]
    console.print(
        Panel.fit(
            f"[bold green]Starting MLX server[/bold green]\n"
            f"Model: {MLX_SERVER_MODEL}\n"
            f"URL:   {MLX_SERVER_BASE_URL}",
            border_style="green",
        )
    )
    return subprocess.Popen(cmd)


def server_is_running() -> bool:
    import requests

    try:
        r = requests.get(f"{MLX_SERVER_BASE_URL}/models", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_server(timeout: float = 180.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if server_is_running():
            return True
        time.sleep(1)
    return False


def voice_loop(
    agent,
    no_voice: bool = False,
    noise_multiplier: float = NOISE_MULTIPLIER,
    min_noise_floor: float = MIN_NOISE_FLOOR,
):
    if no_voice:
        console.print(
            Panel.fit(
                "[bold cyan]Local Text Agent Ready[/bold cyan]\n"
                "Press [bold]Ctrl+C[/bold] to exit.\n"
                "Type your prompt, then hit enter to send.",
                border_style="cyan",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold cyan]Local Voice Agent Ready[/bold cyan]\n"
                "Press [bold]Ctrl+C[/bold] to exit.\n"
                "Speak after the prompt, then pause to send.",
                border_style="cyan",
            )
        )

    from agent_core import record_until_silence, transcribe_audio, speak

    while True:
        try:
            if no_voice:
                text = input("You: ")
                if not text:
                    continue
            else:
                audio = record_until_silence(
                    noise_multiplier=noise_multiplier, min_noise_floor=min_noise_floor
                )
                if len(audio) == 0:
                    continue

                console.print("\U0001f4dd Transcribing...")
                text = transcribe_audio(audio)
                if not text:
                    console.print("[dim]No speech detected.[/dim]")
                    continue
                console.print(f"[bold white]You:[/bold white] {text}")

            if text.lower().strip() in ("exit", "quit", "goodbye", "stop"):
                console.print("[bold yellow]Goodbye![/bold yellow]")
                if not no_voice:
                    speak("Goodbye!")
                break

            console.print("\U0001f914 Thinking...")
            result = agent.run(text)
            response = result.output
            console.print(f"[bold green]Agent:[/bold green] {response}")

            if not no_voice:
                speak(response)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted. Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {type(e).__name__}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Local voice agent powered by MLX + agent_weave"
    )
    parser.add_argument(
        "--server", action="store_true", help="Run the MLX server only."
    )
    parser.add_argument(
        "--auto-server", action="store_true", help="Start the MLX server automatically."
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice input/output (text only mode).",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=NOISE_MULTIPLIER,
        help="Multiplier for the silence threshold.",
    )
    parser.add_argument(
        "--min-noise-floor",
        type=float,
        default=MIN_NOISE_FLOOR,
        help="Minimum noise floor for the silence threshold.",
    )
    args = parser.parse_args()

    if args.server:
        proc = start_mlx_server()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    server_proc = None
    if args.auto_server:
        if server_is_running():
            console.print("[green]MLX server is already running.[/green]")
        else:
            server_proc = start_mlx_server()
            console.print(
                "Waiting for server to become ready (first run " "downloads ~4 GB)..."
            )
            if not wait_for_server():
                console.print("[bold red]Server did not start in time.[/bold red]")
                if server_proc:
                    server_proc.terminate()
                sys.exit(1)
            console.print("[green]Server is ready![/green]")
    else:
        if not server_is_running():
            console.print(
                Panel.fit(
                    "[bold red]MLX server is not running.[/bold red]\n\n"
                    "Start it in another terminal with:\n"
                    f"  [bold]python -m mlx_lm.server --model {MLX_SERVER_MODEL} "
                    f"--port {MLX_SERVER_PORT}[/bold]\n\n"
                    "Or re-run this script with [bold]--auto-server[/bold].",
                    border_style="red",
                )
            )
            sys.exit(1)

    console.print("Building agent...")
    from agent_core import build_agent

    agent = build_agent()

    try:
        voice_loop(
            agent,
            no_voice=args.no_voice,
            noise_multiplier=args.noise_multiplier,
            min_noise_floor=args.min_noise_floor,
        )
    finally:
        if server_proc:
            console.print("Shutting down MLX server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
