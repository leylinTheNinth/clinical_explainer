"""
Clinical Explainer Package for Multiple Choice Medical Questions
"""

# Try to import required packages, if not present, install them
import importlib
import subprocess
import sys

def import_or_install(package_name, version=None):
    try:
        if version:
            package_spec = f"{package_name}>={version}"
        else:
            package_spec = package_name
            
        # Try to upgrade if it's bitsandbytes
        if package_name == 'bitsandbytes':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package_spec])
        else:
            importlib.import_module(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])

# Required packages with versions where needed
required_packages = [
    ('datasets', None),
    ('transformers', None),
    ('torch', None),
    ('lime', None),
    ('shap', None),
    ('tokenshap', None),
    ('bitsandbytes', '0.41.1')  # Specify minimum version for 8-bit quantization
]

for package, version in required_packages:
    import_or_install(package, version)

# Now import the Pipeline
from .pipeline import Pipeline

__version__ = '1.1.1'