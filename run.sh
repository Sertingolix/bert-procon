
#enter virtual env
source ./venv_tf1/bin/activate

#setting env var for models
export BASE_DIR=$PWD

cd code/

#call python handler with all arguments
python run.py "$@"


#cleanup
deactivate