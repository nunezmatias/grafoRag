from setuptools import setup, find_packages

setup(
    name="graphrag_core",
    version="0.1.0",
    description="Agnostic GraphRAG Engine with Node-Centric Retrieval and Causal Traversal",
    author="GraphRAG Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "chromadb",
        "sentence-transformers",
        "networkx",
        "torch",
        "tqdm",
        "pyvis"
    ],
    python_requires=">=3.9",
)