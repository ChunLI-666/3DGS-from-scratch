from setuptools import setup, find_packages

setup(
    name="3dgs-from-scratch",
    version="0.1.0",
    description="Educational implementation of 3D Gaussian Splatting",
    author="3DGS Tutorial Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "viz": [
            "plotly>=5.15.0",
            "pyvista>=0.40.0",
            "ipywidgets>=8.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "jupyterlab>=4.0.0",
        ],
    },
)
