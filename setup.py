import setuptools
import sys

if sys.version_info.major != 3:
    raise TypeError('This Python is only compatible with Python 3, but you are running '
                    'Python {}. The installation will likely fail.'.format(sys.version_info.major))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sipga",  # this is the name displayed in 'pip list'
    version="1.0.0",
    author="Anonymous Authors",
    description="Repository for the Paper: "
                "Successful ingredients of Policy Gradient Algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/successful-ingredients-paper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
