import pathlib

import setuptools

root_directory = pathlib.Path(__file__).parent
long_description = (root_directory / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="ultralytics-yolov5",
    version="0.0.1.dev",
    description="Ultralytics YOLOv5 🚀 Python package, https://ultralytics.com",
    python_requires=">=3.6.2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ultralytics/yolov5",
    author="Ultralytics",
    author_email='glenn.jocher@ultralytics.com',
    license="GPLv3",
    packages=["yolov5"],
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.2.2",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "Pillow",
        "PyYAML>=5.3.1",
        "scipy>=1.4.1",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm>=4.41.0",
        "tensorboard>=2.4.1",
        "seaborn>=0.11.0",
        "pandas"
    ],
    extras_require={
        "tests": [
            "pytest",
        ],
        "export": [
            "coremltools>=4.1",
            "onnx>=1.9.0",
            "scikit-learn>=0.24"
        ],
        "extras": [
            "Cython",
            "pycocotools>=2.0",
            "thop"
        ]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: Windows",
        "Operating System :: Linux",
        "Operating System :: MacOS"
    ],
    project_urls={
        'Bug Reports': 'https://github.com/ultralytics/yolov5/issues',
        'Funding': 'https://ultralytics.com',
        'Source': 'https://github.com/ultralytics/yolov5/',
    },
    entry_points={
        'console_scripts': [
            'yolov5_train=yolov5.train:main',
            'yolov5_test=yolov5.test:main',
            'yolov5_detect=yolov5.detect:main',
            'yolov5_export=yolov5.models.export:main'
        ],
    }
)
