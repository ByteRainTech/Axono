"""
C 扩展模块配置
"""
from setuptools import Extension

ext = Extension(
    name="axono._pseudo",
    sources=["python/axono/_pseudo.c"],
)
