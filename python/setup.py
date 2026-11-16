from setuptools import setup

setup(
    name="turk",
    version="0.1.0",
    py_modules=["turk", "cli"],
    install_requires=[],
    extras_require={"numpy": ["numpy>=1.19.0"]},
    entry_points={"console_scripts": ["turk-py=cli:main"]},
    author="SphereX",
    author_email="turkdilmodeli@gmail.com",
    description="Türkçe Yapay Zeka DSL — Python CLI sarmalayıcısı",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/snnkr/turk",
    python_requires=">=3.8",
)
