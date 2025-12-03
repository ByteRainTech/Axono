# Please use `pyproject.toml` instead of `setup.py`

from pathlib import Path

from setuptools import Extension, find_packages, setup

ext = Extension(  # 加载 C 模块
    name="axono._pseudo",
    sources=[str(Path("python") / "axono" / "_pseudo.c")],
)


# 读取 requirements.txt 文件
def load_requirements(file_name: str):
    return [
        pkg_name.strip()  # 去掉两端的空格
        for pkg_name in (
            # 读取文件内容，按行分割
            Path(file_name).read_text(encoding="utf8").splitlines()
        )
        if (
            pkg_name.strip()  # 处理空行
            and not pkg_name.startswith("#")  # 处理注释行
        )
    ]


root = Path(__file__).parent
lib = root / "python" / "axono" / "library"

print("Root Path:", lib)

lib_files = [
    str(
        lib_path.relative_to(root / "python")  # 转换为相对路径
    ).replace("\\", "/")  # 统一路径分隔符
    for lib_path in lib.rglob("*")  # 遍历所有文件
    if lib_path.suffix in {".dll", ".so", ".pyd"}  # 只保留产物
]

print("Library Files:")
print(lib_files)

data_files = [  # 将产物文件复制到 python/axono/library 目录下
    str(Path("python") / str(f)) for f in lib_files
]

setup(
    name="axono",
    version="0.1.0",
    package_dir={"": "python"},
    packages=find_packages(where="python", include=["axono.library"]),
    python_requires=">=3.8",
    install_requires=load_requirements("requirements.txt"),
    license="Apache-2.0",
    ext_modules=[ext],
    classifiers=[
        # "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    data_files=data_files,  # type: ignore
    include_package_data=True,
    zip_safe=False,
)

print("Data Files:")
for data_path in data_files:
    print("  -", data_path)
