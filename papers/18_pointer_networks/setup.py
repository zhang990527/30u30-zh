from setuptools import setup, find_packages

setup(
    name="day18_pointer_networks",
    version="1.0.0",
    description='30u30 项目的指针网络实现',
    author="30u30 Project",
    author_email="project@30u30.ai",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
