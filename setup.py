from setuptools import find_packages,setup # type: ignore

def get_requirements(file_path):
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]

    return requirements

setup(
name="house_price_project",
version="0.0.1",
author="Tharun",
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)