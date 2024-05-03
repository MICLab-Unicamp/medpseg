import os
import setuptools
from medpseg import __version__, check_weight


# Before anything, check presence of weights
check_weight(os.path.join("medpseg", "poly_medseg_25d_fix.ckpt"))
check_weight(os.path.join("medpseg", "poly_lung.ckpt"))
check_weight(os.path.join("medpseg", "lober.ckpt"))


with open("README.md", "r") as fh:
    long_description = fh.read()

found = setuptools.find_packages()
print(f"Found these packages to add: {found}")

setuptools.setup(
    name="medpseg",
    version=__version__,
    author="Diedre Carmo",
    author_email="diedre@dca.fee.unicamp.br",
    description="Modified EfficientDet for Polymorphic Pulmonary Segmentation (MEDPSeg)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MICLab-Unicamp/medpseg",
    packages=found,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['setuptools', 
                      'numpy', 
                      'rich', 
                      'pillow', 
                      'scipy', 
                      'tqdm', 
                      'torch', 
                      'pandas', 
                      'torchvision', 
                      'pytorch-lightning', 
                      'efficientnet_pytorch', 
                      'connected-components-3d==3.12.6', 
                      'psutil', 
                      'gputil', 
                      'opencv-python', 
                      'SimpleITK==2.0.2', 
                      'pydicom', 
                      'matplotlib',
                      'timm',
                      'torchinfo',
                      'monai==1.3.0',
                      'imageio',
                      'nibabel',
                      'wget',
                      'json2table'],
    entry_points={
        'console_scripts': ["medpseg = medpseg.run:main", "medpseg_cpu = medpseg.run:main_cpu"]
    },
    include_package_data=True,
    package_data={'medpseg': ["assets/icon.png", "assets/icon_original.png", "assets/coronacases_100_003.png", "assets/respiratory.gif", "assets/diseased.gif", "poly_medseg_25d_fix.ckpt", "poly_lung.ckpt", "lober.ckpt"]} # v4 removed all old weight requirements
)
