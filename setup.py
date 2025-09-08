from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name = "snaption",
    version= "0.1.0",
    author = "Aamir Khan",
    author_email = "aamirksfg@gmail.com",
    description = "AI-Powered Image Captioning using PyTorch",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Akhan521/Snaption",
    packages = find_packages(),
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords = "image captioning, ai, deep learning, pytorch, transformers, multimodal, vision-language",
    python_requires = ">=3.12",
    install_requires = requirements,
    entry_points = {
        "console_scripts": [
            "snaption=snaption.cli:main",
        ],
    },
    include_package_data = True,
    zip_safe = False,
)