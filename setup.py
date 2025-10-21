from setuptools import setup, find_packages
from pathlib import Path

# --- Optional Docs command integration ---
try:
    from make_sphinx_documentation import BuildDocsCommand
    BuildDocsCommand.sourcedir = "docs"
    cmdclass = {'build_docs': BuildDocsCommand}
except ImportError:
    cmdclass = {}

setup(
    name="LightCraft",
    version="1.0.0",
    author="Schizo Studios",
    author_email="contact@schizostudios.org",
    description="Light-based metaphysical rendering engine from Schizo Studios.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://schizostudios.org",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "lightcraft=lightcraft.__main__:main",
        ],
    },
    install_requires=[
        "numpy",
        "pillow",
        "imageio",
        "moviepy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    cmdclass=cmdclass,
)