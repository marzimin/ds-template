from setuptools import find_packages, setup

setup(
    name="ds-template",
    version="1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pipeline=ds_template.main:main",
        ],
    },
)
