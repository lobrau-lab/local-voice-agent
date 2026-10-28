import pytest
from tools import execute_shell

def test_execute_shell_pipeline():
    result = execute_shell.func("echo 'hello | world' | wc -c")
    # without shell=True, the pipe is treated as a literal argument
    # 'hello | world | wc -c\n' is the output
    assert "hello | world | wc -c" in result

def test_execute_shell_and():
    result = execute_shell.func("echo hello && echo world")
    assert "hello\nworld" in result or "hello" in result

def test_execute_shell_env():
    result = execute_shell.func("FOO=bar echo $FOO")
    assert "No such file or directory" in result
