# Core dependencies
pip install opencv-python
pip install pyyaml
pip install matplotlib
pip install tqdm
pip install shapely
pip install cython
pip install scikit-image
pip install termcolor
pip install setuptools==59.5.0  # Prevent compatibility issues

# Install PyTorch (CPU version)
pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu \
--extra-index-url https://download.pytorch.org/whl/cpu \
--trusted-host download.pytorch.org \
--trusted-host pypi.org \
--trusted-host files.pythonhosted.org