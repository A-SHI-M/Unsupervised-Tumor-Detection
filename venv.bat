@echo off

curl https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip -o python-3.12.10-embed-amd64.zip

mkdir python-3.12.10

move python-3.12.10-embed-amd64.zip python-3.12.10

cd python-3.12.10

tar   -xf python-3.12.10-embed-amd64.zip

del python-3.12.10-embed-amd64.zip

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

.\python.exe get-pip.py

del get-pip.py

echo import site >> python312._pth

.\Scripts\pip.exe install virtualenv

cd ..

python-3.12.10\scripts\virtualenv.exe -p python-3.12.10\python.exe venv

venv\scripts\activate



