import subprocess
result = subprocess.run(["bash", "-c", "echo hello && echo world"], capture_output=True, text=True)
print(repr(result.stdout))
