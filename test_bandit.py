import subprocess

def execute_shell_bash(command: str):
    subprocess.run(["bash", "-c", command], shell=False)
