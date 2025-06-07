# setup.py
from setuptools import setup, find_packages

setup(
    name="subgroups",
    version="0.1.0",
    packages=find_packages(),       # <— tells setuptools to include `subgroups` and sub‑packages
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "chz", 
        "xgboost"
        # …any other runtime deps…
    ],
    author="Your Name",
    description="My subgroups analysis package",
    include_package_data=True,      # if you have package_data
)
