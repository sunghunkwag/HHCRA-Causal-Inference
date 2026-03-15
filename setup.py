from setuptools import setup, find_packages

setup(
    name="coda-causal",
    version="1.0.0",
    description="CODA: Cross-validated Ordering for DAG Alignment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sunghun Kwag",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
)
