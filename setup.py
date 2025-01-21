from setuptools import setup, find_packages

setup(
    name="my_package",  # Replace with your package's name
    version="0.1.0",    # Replace with your version
    package_dir={"": "src"},  # Specifies that packages are under the `src` directory
    packages=find_packages(where="src"),  # Automatically find all packages in `src`
    install_requires=[],  # Add dependencies here if needed
)