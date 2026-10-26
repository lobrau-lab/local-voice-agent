import subprocess

def execute_shell_bash(command: str):
    subprocess.run(["/bin/bash", "-c", command], shell=False)
