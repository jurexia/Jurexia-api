
from google.genai import types
import inspect

classes = [t for t in dir(types) if 'Tuned' in t or 'Tuning' in t or 'Hyper' in t]
for cls_name in classes:
    print(f"Class: {cls_name}")
    cls = getattr(types, cls_name)
    try:
        # Check if it has an __init__ or similar
        print(f"  Init: {inspect.signature(cls.__init__)}")
    except:
        pass
