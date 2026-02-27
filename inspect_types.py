
from google.genai import types
import inspect

print("Types in google.genai.types:")
for name, obj in inspect.getmembers(types):
    if name.startswith('Tuned'):
        print(name)
