from setuptools import setup, find_packages

setup(
    name="luckyrobot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "pybullet>=3.2.5",
        "tianshou>=0.4.11",
        "gymnasium>=0.26.0",
        "opencv-python>=4.5.3",
        "wandb>=0.12.0",
    ],
) 