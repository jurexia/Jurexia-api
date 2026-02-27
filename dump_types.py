
from google.genai import types
import json

data = {}
for name in dir(types):
    if not name.startswith('_'):
        data[name] = str(getattr(types, name))

with open('sdk_types.json', 'w') as f:
    json.dump(data, f, indent=2)
