#!/usr/bin/env python3
"""
Setup script for DataMCPServerAgent.
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="datamcpserveragent",
    version="0.1.0",
    author="Dima",
    author_email="aws.inspiration@gmail.com",
    description="Advanced agent architectures for Bright Data MCP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DimaJoyti/DataMCPServerAgent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "datamcpserveragent=src.core.main:chat_with_agent",
            "datamcpserveragent-advanced=src.core.advanced_main:chat_with_advanced_agent",
            "datamcpserveragent-enhanced=src.core.enhanced_main:chat_with_enhanced_agent",
            "datamcpserveragent-advanced-enhanced=src.core.advanced_enhanced_main:chat_with_advanced_enhanced_agent",
            "datamcpserveragent-multi-agent=src.core.multi_agent_main:chat_with_multi_agent_learning_system",
            "datamcpserveragent-rl=src.core.reinforcement_learning_main:chat_with_rl_agent",
            "datamcpserveragent-distributed=src.core.distributed_memory_main:chat_with_distributed_memory_agent",
            "datamcpserveragent-knowledge-graph=src.core.knowledge_graph_main:chat_with_knowledge_graph_agent",
            "datamcpserveragent-seo=src.core.seo_main:chat_with_seo_agent",
        ],
    },
)
