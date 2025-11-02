from pathlib import Path
from setuptools import setup, find_packages

def load_req(fn: str):
    return [r.strip() for r in Path(fn).read_text(encoding="utf8").splitlines()
            if r.strip() and not r.startswith("#")]

setup(
    name="axono",
    version="0.1.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.8",
    install_requires=load_req("requirements.txt"),
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_data={
        "axono": ["library/*.so", "library/*.dll", "library/*.dylib"],
    },
    include_package_data=True,
    zip_safe=False,
)
