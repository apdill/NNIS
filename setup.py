from setuptools import setup, find_packages

setup(
    name='NNIS', 
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'opencv-python', 
        'tifffile',
    ],
)