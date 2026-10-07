from setuptools import setup

setup(
    name="mathbot",
    version="0.1.0",
    author="Sadegh Karami Chefteh",
    description="Symbolic discovery of implicit manifold equations",
    url="https://github.com/sadeghkarami/mathbot",
    py_modules=["mathbot", "mathbot_core", "discover",
                "freq_lib", "denoise", "saver", "make_sample"],
    python_requires=">=3.8",
    install_requires=["numpy>=1.20", "scipy>=1.7", "sympy>=1.10"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
