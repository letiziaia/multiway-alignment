from setuptools import setup, find_packages  # type: ignore

setup(
    name="multiway_alignment",
    version="0.0.1",
    packages=find_packages(),
    author="Letizia Iannucci",
    author_email="letizia.iannucci@aalto.fi",
    description="Quantifying multiway (higher-order) alignment with mutual information",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/letiziaia/multiway-alignment",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
