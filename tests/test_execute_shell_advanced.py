import pytest
from tools import execute_shell

def test_execute_shell_pipeline():
    result = execute_shell.func("echo 'hello | world' | wc -c")
    # 'hello | world\n' is 14 characters
    assert "14" in result

def test_execute_shell_and():
    result = execute_shell.func("echo hello && echo world")
    assert "hello\nworld" in result or "hello" in result

def test_execute_shell_env():
    result = execute_shell.func("FOO=bar echo $FOO")
    assert "bar" in result
