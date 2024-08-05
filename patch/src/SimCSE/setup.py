import io

from setuptools import setup

with io.open("./README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="simcse",
    packages=["simcse"],
    version="0.4",
    license="MIT",
    description="A sentence embedding tool based on SimCSE",
    author="Tianyu Gao, Xingcheng Yao, Danqi Chen",
    author_email="tianyug@cs.princeton.edu",
    url="https://github.com/princeton-nlp/SimCSE",
    download_url="https://github.com/princeton-nlp/SimCSE/archive/refs/tags/0.4.tar.gz",
    keywords=["sentence", "embedding", "simcse", "nlp"],
    install_requires=["tqdm", "scikit-learn", "scipy", "transformers", "torch", "numpy>=1.20.0,<2.0.0", "setuptools"],
)
