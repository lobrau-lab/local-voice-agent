1. **Understand & Measure**: Analyzed the problem and benchmarked the unoptimized code (`_is_path_allowed` checking 10k times took ~5.60 seconds).
2. **Implement**: Optimized `_is_path_allowed` in `tools.py` by caching the resolved values of `ALLOWED_ROOTS`. Handled dynamic invalidation by verifying tuple match.
3. **Verify**: The benchmark time went from ~5.60 seconds to ~2.44 seconds. Tests were run via `python3 -m pytest tests/test_tools.py` and passed successfully.
4. **Pre-commit**: I am calling `pre_commit_instructions` now to prepare for submission.
5. **Submit**: Submitting the PR via agent weave framework.
