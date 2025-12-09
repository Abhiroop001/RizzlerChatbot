import sys
import importlib
import pkg_resources

REQUIRED_PACKAGES = {
    "Flask": None,
    "flask_cors": "flask-cors",
    "gunicorn": None,
    "numpy": None,
    "nltk": None,
    "tensorflow": "tensorflow-cpu",
    "pickle5": None
}

def check_python_version():
    print("🔧 Python Version Check")
    print("------------------------")
    print(f"Running Python: {sys.version}\n")
    if sys.version_info < (3, 9):
        print("⚠️  Warning: Python 3.9+ recommended for TensorFlow.")
    print()


def check_packages():
    print("📦 Package Requirements Check")
    print("------------------------------")

    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "Unknown")
            print(f"✓ {module_name} installed (version: {version})")
        except ImportError:
            pkg_to_install = pip_name if pip_name else module_name
            print(f"✗ {module_name} NOT installed")
            missing.append(pkg_to_install)

    if missing:
        print("\n❌ Missing packages detected!")
        print("Install them using:")
        print("\npip install " + " ".join(missing))
    else:
        print("\n✅ All required packages are installed!")

    print()


def check_tensorflow():
    print("🧠 TensorFlow Check")
    print("--------------------")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        physical = tf.config.list_physical_devices("CPU")
        print(f"CPU Devices: {physical}")

    except ImportError:
        print("❌ TensorFlow not installed.")
    except Exception as e:
        print(f"⚠️ TensorFlow error: {e}")

    print()


if __name__ == "__main__":
    print("=== Environment Verification Script ===\n")
    check_python_version()
    check_packages()
    check_tensorflow()
    print("✔ Environment check completed.")
