from setuptools import setup, find_packages


setup(
    name="speech_utils",
    version="0.0.1",
    author="Hoang Nghia Tuyen",
    author_email="hnt4499@gmail.com",
    url="https://github.com/NTU-SER/speech_utils",
    install_requires=["numpy"],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.md"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="A utilities package for speech processing",
    keywords=["deep learning", "natural language processing",
              "natural language undestanding", "speech emotion recognition"]
)
