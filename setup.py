from setuptools import setup, find_packages

setup(
    name="speaker_clustering",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "librosa",
        "pandas",
        "numpy",
        "scikit-learn",
        "speechbrain",
    ],
    entry_points={
        "console_scripts": ["speaker-clustering=scripts.run_clustering:main"]
    },
)
