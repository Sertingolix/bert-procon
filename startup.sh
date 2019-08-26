pip3 install --user pipenv
virtualenv --system-site-packages -p python3 ./venv_tf1
source ./venv_tf1/bin/activate
pip3 install absl-py==0.7.1
pip3 install tensorflow-gpu==1.13.1
pip3 install Scrapy
pip3 install beautifulsoup4

deactivate
