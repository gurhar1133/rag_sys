from setuptools import setup, find_packages

setup(
    name="simple-rag",
    version="0.1.0",
    description="A minimal Retrieval-Augmented Generation system",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "faiss-cpu",
        "numpy",
        "tqdm",
        "pandas",
        "pathlib",
        "torch",
        "sentence_transformers",
    ],
    python_requires=">=3.9",
)