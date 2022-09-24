pip3 install --upgrade pip

echo "Installing virtualenv"
pip3 install virtualenv

echo "virtualenv version:"
virtualenv --version

# change python 3.7 path according
echo "Creating virtual environment"
virtualenv -p /Library/Frameworks/Python.framework/Versions/3.7/bin/python3 aicadium_coding

# echo "ACTIVATING VENV"
# source aicadium_coding/bin/activate
