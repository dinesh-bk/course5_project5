import subprocess

# Run a command and wait for it to complete
result = subprocess.run(['ls', '-l'], capture_output=True, text=True)

# Output the command's stdout and stderr
print('stdout:', result.stdout)
#print('stderr:', result.stderr)
print('Return code:', result.returncode)