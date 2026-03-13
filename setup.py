from setuptools import setup, find_packages

setup(
    name="hhcra",
    version="0.5.0",
    description="Hierarchical Hybrid Causal Reasoning Architecture",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
