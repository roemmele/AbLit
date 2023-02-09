import setuptools

setuptools.setup(
    name="ablit",
    version="1.0.0",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    install_requires=[],
    include_package_data=True
)
