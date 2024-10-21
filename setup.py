# setup.py

from setuptools import setup, find_packages

setup(
    name="NNIS",  # Consider using lowercase 'nnis' for best practices
    version="0.1.0",
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=2.2.3",
        "pytz>=2020.1",
        "matplotlib",
        "opencv-python",
        "pillow",
        "tifffile",
        # Add other dependencies as needed
    ],
    author="Aidem Dillon",
    description="A package for neuronal network generation and instance segmentation",
    url="https://github.com/apdill/NNIS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update if different
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Update if you require a different version
)
