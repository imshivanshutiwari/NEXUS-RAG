from setuptools import setup, find_packages

setup(
    name="nexus-rag",
    version="1.0.0",
    description="Production-Grade Multi-Modal Agentic RAG System with Self-Healing Pipelines",
    author="NEXUS Team",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "nexus-run=dashboard.app:main",
            "nexus-api=api.server:main",
            "nexus-ingest=ingestion.pipeline:main",
        ]
    },
)
