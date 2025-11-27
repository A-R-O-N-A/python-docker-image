import os
import cowsay

import subprocess
import sys


print('Hello to docker')
print('I hope i can learn docker with python')
print('The current directory is : ', os.getcwd())

print('This is a new change')

cowsay.cow('Good Moooooooooooooorning !')

# Test Poetry (optional)
try:
    # Run poetry --version in a subprocess
    result = subprocess.run([sys.executable, "-m", "poetry", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Poetry is installed:", result.stdout.strip())
    else:
        print("Poetry not found.")
except Exception as e:
    print("Error checking Poetry:", e)
