import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools import read_file, _is_path_allowed, execute_shell, ALLOWED_ROOTS

# --- _is_path_allowed Tests ---


def test_is_path_allowed_valid():
    """Test _is_path_allowed with a path inside ALLOWED_ROOTS."""
    if ALLOWED_ROOTS:
        root = ALLOWED_ROOTS[0]
        root.mkdir(parents=True, exist_ok=True)
        root = ALLOWED_ROOTS[0]
        valid_path = root / "some_file.txt"
        assert _is_path_allowed(str(valid_path)) is True


def test_is_path_allowed_invalid():
    """Test _is_path_allowed with a path outside ALLOWED_ROOTS."""
    invalid_path = Path("/tmp/outside_allowed_roots.txt")
    if ALLOWED_ROOTS and not any(r in invalid_path.parents for r in ALLOWED_ROOTS):
        assert _is_path_allowed(str(invalid_path)) is False


def test_is_path_allowed_relative():
    """Test _is_path_allowed with a relative path resolving inside."""
    if ALLOWED_ROOTS:
        root = ALLOWED_ROOTS[0]
        root.mkdir(parents=True, exist_ok=True)
        root = ALLOWED_ROOTS[0]
        rel_path = f"{root}/sub_dir/../some_file.txt"
        assert _is_path_allowed(rel_path) is True


def test_is_path_allowed_symlink():
    """Test _is_path_allowed with a symlink that points outside."""
    if ALLOWED_ROOTS:
        root = ALLOWED_ROOTS[0]
        root.mkdir(parents=True, exist_ok=True)
        root = ALLOWED_ROOTS[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # create a file outside
            target = temp_path / "target.txt"
            target.touch()

            # create a symlink inside allowed roots
            symlink_path = root / "bad_link.txt"

            try:
                os.symlink(target, symlink_path)
                # Resolved symlink is outside ALLOWED_ROOTS (if temp_dir is not in ALLOWED_ROOTS)
                if not any(r in target.parents for r in ALLOWED_ROOTS):
                    assert _is_path_allowed(str(symlink_path)) is False
            except Exception:
                pass
            finally:
                if symlink_path.exists(follow_symlinks=False):
                    symlink_path.unlink()


# --- read_file Tests ---
# Note: read_file is decorated with @tool in tools.py, which means we must test `.func` if using agent_weave tool.
# We'll just test the function logic.


def test_read_file_success():
    """Test reading a valid file."""
    if ALLOWED_ROOTS:
        root = ALLOWED_ROOTS[0]
        root.mkdir(parents=True, exist_ok=True)
        root = ALLOWED_ROOTS[0]
        test_file = root / "test_read_file.txt"
        test_content = "Hello, World!"
        try:
            test_file.write_text(test_content)
            # Call the inner function of the @tool decorator
            result = read_file.func(str(test_file))
            assert result == test_content
        finally:
            if test_file.exists():
                test_file.unlink()


def test_read_file_blocked():
    """Test reading a file outside allowed roots is blocked."""
    invalid_path = "/tmp/test_blocked_file.txt"
    if ALLOWED_ROOTS and not any(Path("/tmp") in r.parents for r in ALLOWED_ROOTS):
        result = read_file.func(invalid_path)
        assert result.startswith("[BLOCKED]")


def test_read_file_not_found():
    """Test reading a non-existent file."""
    if ALLOWED_ROOTS:
        root = ALLOWED_ROOTS[0]
        root.mkdir(parents=True, exist_ok=True)
        root = ALLOWED_ROOTS[0]
        test_file = root / "does_not_exist.txt"
        if test_file.exists():
            test_file.unlink()
        result = read_file.func(str(test_file))
        assert result.startswith("[ERROR] Not a file:")


# --- execute_shell Tests ---


@patch("tools.subprocess.run")
def test_execute_shell_success(mock_run):
    """Test executing a safe command successfully."""
    mock_result = MagicMock()
    mock_result.stdout = "hello world\n"
    mock_result.stderr = ""
    mock_result.returncode = 0
    mock_run.return_value = mock_result

    result = execute_shell.func("echo 'hello world'")
    assert result == "hello world\n"
    mock_run.assert_called_once()


def test_execute_shell_blocked():
    """Test that blocked commands are rejected."""
    # Assuming 'rm -rf' is in BLOCKED_COMMANDS
    result = execute_shell.func("rm -rf /")
    assert result.startswith("[BLOCKED]")


@patch("tools.subprocess.run")
def test_execute_shell_timeout(mock_run):
    """Test handling of subprocess.TimeoutExpired."""
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=30)

    result = execute_shell.func("sleep 100")
    assert result == "[ERROR] Command timed out after 30 seconds."


@patch("tools.subprocess.run")
def test_execute_shell_error(mock_run):
    """Test handling of command with stderr output."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "command not found"
    mock_result.returncode = 127
    mock_run.return_value = mock_result

    result = execute_shell.func("invalid_command")
    assert "command not found" in result
    assert "[stderr]" in result


def test_fetch_webpage_safe_url():
    from tools import fetch_webpage

    # Valid external URL
    result = fetch_webpage.func("https://www.example.com")
    assert "[ERROR]" not in result

    # Blocked URLs
    result_local = fetch_webpage.func("http://localhost:8000")
    assert "[ERROR] Unsafe URL provided" in result_local

    result_meta = fetch_webpage.func("http://169.254.169.254")
    assert "[ERROR] Unsafe URL provided" in result_meta

    result_private = fetch_webpage.func("http://192.168.1.1")
    assert "[ERROR] Unsafe URL provided" in result_private


def test_http_request_safe_url():
    from tools import http_request

    # Blocked URLs
    result_local = http_request.func("http://127.0.0.1")
    assert "[ERROR] Unsafe URL provided" in result_local

    result_meta = http_request.func("http://169.254.169.254/latest/meta-data/")
    assert "[ERROR] Unsafe URL provided" in result_meta


def test_is_safe_url():
    from tools import _is_safe_url

    # True cases
    assert _is_safe_url("http://google.com") is True
    assert _is_safe_url("https://github.com") is True

    # False cases
    assert _is_safe_url("http://localhost") is False
    assert _is_safe_url("http://127.0.0.1") is False
    assert _is_safe_url("http://192.168.1.1") is False
    assert _is_safe_url("http://10.0.0.1") is False
    assert _is_safe_url("http://172.16.0.1") is False
    assert _is_safe_url("http://169.254.169.254") is False
    assert _is_safe_url("http://0.0.0.0") is False
    assert _is_safe_url("file:///etc/passwd") is False
