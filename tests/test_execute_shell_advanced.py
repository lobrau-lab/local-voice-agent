import pytest
from tools import execute_shell


def test_execute_shell_pipeline():
    result = execute_shell.func("echo 'hello | world' | wc -c")
    # With shell=False, pipelines are passed as literals to echo
    assert "hello | world | wc -c" in result


def test_execute_shell_and():
    result = execute_shell.func("echo hello && echo world")
    # With shell=False, logical operators are passed as literals to echo
    assert "hello && echo world" in result


def test_execute_shell_env():
    result = execute_shell.func("export FOO=bar")
    # With shell=False, export command does not exist
    assert "FileNotFoundError" in result or "No such file or directory" in result
