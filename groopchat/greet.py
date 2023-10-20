# filename: greet.py
import sys

first_name = sys.argv[1]
last_name = sys.argv[2]

full_name = f"{first_name} {last_name}"
print("Hello", full_name)