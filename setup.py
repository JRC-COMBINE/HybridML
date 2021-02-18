import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HybridML",
    version="0.0.1",
    author="Younes Mueller",
    author_email="ymueller@aices.rwth-aachen.de",
    description="Build hybrid tensorflow models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://www.combine.rwth-aachen.de/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
    ],
    python_requires=">=3.6",
)
