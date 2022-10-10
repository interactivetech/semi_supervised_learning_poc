$pwd
eval "$(conda shell.bash hook)" && conda activate base 
conda env create -f env.yaml
conda activate usb
