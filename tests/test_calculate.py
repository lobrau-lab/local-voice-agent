import pytest
import sys
import os

from tools import calculate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import calculate


def run_calculate(expression):
    if hasattr(calculate, "func"):
        return calculate.func(expression)
    raise ValueError(f"Could not figure out how to call Tool object.")


def test_calculate_basic():
    """Test basic arithmetic operations."""
    assert run_calculate("2 + 2") == "2 + 2 = 4"
    assert run_calculate("10 - 5") == "10 - 5 = 5"
    assert run_calculate("3 * 4") == "3 * 4 = 12"
    assert run_calculate("20 / 4") == "20 / 4 = 5.0"


def test_calculate_advanced():
    """Test advanced arithmetic and precedence."""
    assert run_calculate("2 ** 3") == "2 ** 3 = 8"
    assert run_calculate("10 % 3") == "10 % 3 = 1"
    assert run_calculate("(2 + 3) * 4") == "(2 + 3) * 4 = 20"
    assert run_calculate("10 / (2 + 3)") == "10 / (2 + 3) = 2.0"
    assert run_calculate("2 + 3 * 4") == "2 + 3 * 4 = 14"


def test_calculate_spacing():
    """Test expressions with varying spacing."""
    assert run_calculate("2+2") == "2+2 = 4"
    assert run_calculate(" 2 + 2 ") == " 2 + 2  = 4"
    assert run_calculate("  10  /  2  ") == "  10  /  2   = 5.0"


def test_calculate_disallowed_chars():
    """Test that execution is blocked for disallowed characters."""
    assert (
        run_calculate("import os")
        == "[ERROR] Expression contains disallowed characters."
    )
    assert (
        run_calculate("eval('2+2')")
        == "[ERROR] Expression contains disallowed characters."
    )
    assert (
        run_calculate("__import__('os').system('ls')")
        == "[ERROR] Expression contains disallowed characters."
    )
    assert (
        run_calculate("2 + a") == "[ERROR] Expression contains disallowed characters."
    )
    assert (
        run_calculate("10 / 2 = 5")
        == "[ERROR] Expression contains disallowed characters."
    )


def test_calculate_errors():
    """Test correct exception handling for invalid expressions."""
    assert "ZeroDivisionError" in run_calculate("1 / 0")
    assert "SyntaxError" in run_calculate("2 + ")
    assert "SyntaxError" in run_calculate("* 5")
    assert "SyntaxError" in run_calculate("10 / / 2")


def test_calculate_edge_cases():
    """Test edge cases such as empty strings and decimals."""
    assert "SyntaxError" in run_calculate("")
    assert "SyntaxError" in run_calculate(" ")
    assert run_calculate("0") == "0 = 0"
    assert run_calculate(".5 + .5") == ".5 + .5 = 1.0"
    assert run_calculate("1.") == "1. = 1.0"
    assert run_calculate("()") == "() = ()"
