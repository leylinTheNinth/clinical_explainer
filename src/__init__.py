"""
Clinical Explainer Package for Multiple Choice Medical Questions
"""

import importlib
import subprocess
import sys

def import_or_install(package_name, version=None):
    try:
        if version:
            package_spec = f"{package_name}>={version}"
        else:
            package_spec = package_name

        # Attempt to import the package
        return importlib.import_module(package_name)
    except ImportError:
        # If import fails, install the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        # Try importing again after installation
        return importlib.import_module(package_name)
    
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