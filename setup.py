from setuptools import setup, find_packages

setup(
    name="graphcast_xai",
    version="0.1.0",
    description="GraphCast XAI utilities",
    packages=find_packages(include=["pkg", "pkg.*"]),
    python_requires="==3.12.10",
)
