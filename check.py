import importlib
import tensorflow as tf
import nltk

def show_version(name, attr="__version__"):
    try:
        module = importlib.import_module(name)
        v = getattr(module, attr, "UNKNOWN")
        print(f"{name:15s} -> {v}")
    except Exception as e:
        print(f"{name:15s} -> ERROR: {e}")

show_version("flask")
show_version("gunicorn")
show_version("numpy")
show_version("scipy")
show_version("sklearn")      # scikit-learn
show_version("pandas")
print(f"{'tensorflow':15s} -> {tf.__version__}")
print(f"{'nltk':15s} -> {nltk.__version__}")