from setuptools import setup, find_packages

setup(
    name="FixedPointJAX",
    version="0.0.20",
    packages=find_packages(),
    install_requires=[
        # List of package dependencies
        'jax',
    ],
    author="Esben Scriver Andersen",
    author_email="esbenscriver@gmail.com",
    description="Fixed-point iterations for root finding implemented in JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/esbenscriver/FixedPointJAX.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)