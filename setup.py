import os
from setuptools import setup
from yolov5 import __version__


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "rb") as fid:
        return fid.read().decode("utf-8")


req = read("requirements.txt").splitlines()

requirements = req + ["setuptools"]

setup(
    name="yolov5",
    version=__version__,
    author="Psycle Research",
    description="Fork of yolov5",
    url="https://github.com/PsycleResearch/yolov5",
    packages=[
        "yolov5",
        "yolov5.models",
        "yolov5.utils",
        "yolov5.utils.loggers",
        "yolov5.utils.loggers.wandb",
    ],
    package_data={
        "yolov5": ["data/*", "data/**/*", "models/*.yaml", "models/**/*.yaml"]
    },
    install_requires=requirements,
    python_requires=">=3.6",
)
