"""
Clinical Explainer Package for Multiple Choice Medical Questions
"""

# Try to import required packages, if not present, install them
import importlib

def import_or_install(package_name):
    try:
        importlib.import_module(package_name)
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Required packages
required_packages = ['datasets', 'transformers', 'torch', 'lime', 'shap']
for package in required_packages:
    import_or_install(package)

# Now import the Pipeline
from .pipeline import Pipeline

__version__ = '1.0.0'
