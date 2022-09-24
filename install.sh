echo "INSTALLING DEPENDENCIES"
python3 -m pip install --upgrade setuptools
pip3 install -r requirements.txt

pip3 install ipykernel
python3 -m ipykernel install --user --name=aicadium_coding