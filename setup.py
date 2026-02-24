from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="visqa",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="Query-driven video segmentation and grounding using open-source models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourname/visqa",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.0.0",
        "transformers>=4.35.0",
        "open_clip_torch>=2.20.0",
        "pycocotools>=2.0.6",
        "supervision>=0.17.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "huggingface_hub>=0.19.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "demo": ["gradio>=4.0.0"],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
        ],
        "all": [
            "gradio>=4.0.0",
            "wandb>=0.16.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "visqa=visqa.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
