import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quanttree",
    version="0.0.4",
    description="Implementation of the QuantTree algorithm and extensions.",
    license="LICENSE.pdf",
    package_dir={"": "."},
    author="Stucchi Diego",
    author_email="stucchidiego1994@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diegocarrera89/quantTree",
    project_urls={
        "Bug Tracker": "https://github.com/diegocarrera89/quantTree/issues",
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)