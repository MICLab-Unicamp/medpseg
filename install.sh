# An optional install script that does all installation steps
echo "Creating medpseg environment..."
eval "$(conda shell.bash hook)"
conda create --name medpseg python=3.9
conda activate medpseg
echo "Installing PyTorch with GPU support..."

# Note this install torch with cuda 11.3. If you need to use a different CUDA version due to your specific GPU, please do it and remove this line.
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo "Installing requirements..."
pip install -r medpseg/requirements.txt
echo "Installing MEDPSeg..."
pip install .
echo "Done. Please always activate the MEDPSeg environment with 'conda activate medpseg' before running MEDPSeg with the 'medpseg' command line command"
medpseg
