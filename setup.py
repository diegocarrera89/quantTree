import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# data_files = []
# roots = ['quanttree/thresholds', 'demos', 'testing']
# for root in roots:
#     for path, subdirs, files in os.walk(root):
#         if not len(files) == 0:
#             data_files.append((path, files))

# for element in data_files:
#     print(element[0])
#     for subelem in element[1]:
#         print(f"    {subelem}")
#
# exit(0)

setuptools.setup(
    name="quanttree",
    version="0.0.3",
    description="Implementation of the QuantTree algorithm and extensions.",
    license_file="LICENSE.pdf",
    package_dir={"": "."},
    packages=['quanttree'],
    # data_files=data_files,
    # packages=setuptools.find_packages(where="."),
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
    python_requires=">=3.6",
)
