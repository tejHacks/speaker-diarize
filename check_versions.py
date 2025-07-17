import pkg_resources
import sys

packages = [
    'streamlit', 'librosa', 'numpy', 'tensorflow', 'soundfile',
    'matplotlib', 'scikit-learn', 'scipy', 'protobuf'
]

print("Installed package versions:")
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"{pkg}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{pkg}: Not installed")

print(f"\nPython version: {sys.version}")