
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

try:
    print("Attempting to import perception.core.pipeline...")
    from perception.core.pipeline import main
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
