from setuptools import setup, find_packages

setup(
    name="ab-testing-experimentation-platform",
    version="1.0.0",
    description="Enterprise Experimentation Engine: CUPED Variance Reduction, Sequential SPRT & SRM Diagnostics",
    author="Siddiqui Zakir Ali",
    author_email="zak.siddiqui16@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "statsmodels>=0.14.0",
    ],
)
