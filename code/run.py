'''
This file handles all flags and prepaires/downloads datasets or checkpoints
'''


import time, os, subprocess, argparse, json
from  run_with_mapping import run_with_mapping
from run_multiple_epochs import run_multiple_epochs


def delte_checkpoints(DIR):
    for root, dirs, files in os.walk(DIR, topdown=True):
        for name in files:
            file = os.path.join(root, name)
            if '.data-00000-of-00001' in file:
                print(file)
                os.remove(file)



BASE_DIR = os.environ['BASE_DIR']
os.chdir(BASE_DIR)

BERT_CHECKPOINT_URL = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'
CHECHPOINT_NAME = 'multi_cased_L-12_H-768_A-12'

BERT_THESIS_CHECKPOINT = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
BERT_THESIS_CHECKPOINT_NAME = 'uncased_L-12_H-768_A-12'

PROCON_PATH = './data/procon-ds-master/output/'
PROCON_BODY_BODY_PATH = PROCON_PATH +'body_body/'
PROCON_QUESTION_BODY_PATH = PROCON_PATH +'question_body/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=4)
    parser.add_argument('--output_dir', help='output directory (default: \'./output/\')',type=str, default='./output')
    parser.add_argument('--dropout', help='dropout for activation and hidden layers (default: 0.1)',type=float, default=0.1)
    parser.add_argument('--bert_dir', help=f'pretrained BERT checkpoint (default: \'./data/{CHECHPOINT_NAME}/\') downloaded if not existing',type=str, default=f'./data/{CHECHPOINT_NAME}/')
    parser.add_argument('--input_dir', help=f'body-body dataset input directory (default: \'{PROCON_BODY_BODY_PATH}\') downloaded if not existing',type=str, default=PROCON_BODY_BODY_PATH)
    parser.add_argument('--qb_input_dir', help=f'question-body dataset input directory (default: \'{PROCON_QUESTION_BODY_PATH}\') downloaded if not existing, only used if --two_step==True',type=str, default=PROCON_QUESTION_BODY_PATH)
    parser.add_argument('--lr', help='learning rate',type=float, default=3e-5)
    parser.add_argument('--two_step',  type=bool, default=False, help='specifies if prediction is in two steps with (default: False)')
    parser.add_argument('--arg_only',  type=bool, default=False, help='only argument is used for training in two_step (default: False)')
    parser.add_argument('--rec',  type=bool, default=True, help='uses the recomended BERT model instead of the one used in thesis(default: True)')

    args = parser.parse_args()

    #loading arguments and process them (like absolute system path)
    N_EPOCH = args.epochs
    DROPOUT = args.dropout
    LEARNING_RATE = args.lr
    RUN_ID = str(int(time.time())) #current time

    #setting correct checkpoint name
    if not args.rec:
        #tesis model used
        if args.bert_dir == f'./data/{CHECHPOINT_NAME}/':
            #bert dir args by default is recomended
            #change bert dir to thesis bert dir
            BERT_DIR = os.path.abspath(f'./data/{BERT_THESIS_CHECKPOINT_NAME}/')+'/'

        BERT_CHECKPOINT_URL = BERT_THESIS_CHECKPOINT
        CHECHPOINT_NAME = BERT_THESIS_CHECKPOINT_NAME
    else:
        #recomended args    
        BERT_DIR = os.path.abspath(args.bert_dir)+'/'

    OUTPUT_DIR = os.path.abspath(args.output_dir)+'/'
    BODY_BODY_INPUT_DIR = os.path.abspath(args.input_dir)+'/'
    QUESTION_BODY_INPUT_DIR = os.path.abspath(args.qb_input_dir)+'/'

    #create name for run
    if args.two_step:
        if args.arg_only:
            task = 'TWO_STEP_WITH_QUESTION'
        else:
            task = 'TWO_STEP_WITHOUT_QUESTION'
    else:
        #plain body body
        task = 'BODY_BODY_PLAIN'

    lr_string = str(LEARNING_RATE).replace('-','')
    RUN_NAME = f'{task}_DROPOUT{int(DROPOUT*10)}_LR{lr_string}_EPOCHS{N_EPOCH}_ID{RUN_ID}/'
    RUN_DIR = OUTPUT_DIR + RUN_NAME
    print(f'RUN_NAME: {RUN_DIR}')

    print('Checking if Checkpoint exists')
    if not os.path.exists(f'{BERT_DIR}/'):
        print('Pretrained BERT checkpoint does not exist')
        print('Downloading BERT checkpoint')

        #downloading and unzipping
        out = subprocess.check_output(["curl", "-o","bert.zip",BERT_CHECKPOINT_URL])
        out = subprocess.check_output(["unzip", "bert.zip",'-d','./data/'])
        os.remove('bert.zip')

    print('Checking if Dataset exists')
    if not os.path.exists(f'{BODY_BODY_INPUT_DIR}/'):
        #clone git repo for dataset
        out = subprocess.check_output(["git", "clone",'https://github.com/Sertingolix/procon-dataset.git','./data/procon-dataset'])
        out = subprocess.check_output(["bash", "run.sh"],cwd="./data/procon-dataset/")
    else:
        print(f'dataset exists in: {BODY_BODY_INPUT_DIR}')
    
    print('setting BERT dropout')
    #loading BERT description File and write new one with right dropout
    #load json 
    with open(BERT_DIR+'bert_config.json') as json_file:
        data = json.load(json_file)

    #edit dropout
    data['attention_probs_dropout_prob']=DROPOUT
    data['hidden_dropout_prob']=DROPOUT

    #write back
    BERT_CONFIG_DIR = BERT_DIR+'bert_config_edited.json'
    with open(BERT_CONFIG_DIR, 'w') as outfile:
        json.dump(data, outfile)


    #plain BERT or two step prediction
    if args.two_step == True:
        print('Two step prediction chosen')
        WITH_QUESTION = not args.arg_only

        run_with_mapping(N_EPOCH=N_EPOCH, LEARNING_RATE=LEARNING_RATE, BERT_DIR=BERT_DIR, BODY_BODY_INPUT_DIR=BODY_BODY_INPUT_DIR, \
            QUESTION_BODY_INPUT_DIR = QUESTION_BODY_INPUT_DIR, OUTPUT_DIR=RUN_DIR, BERT_CONFIG_DIR=BERT_CONFIG_DIR, MAX_SEQUENCE_LENGTH=384,\
            WITH_QUESTION = WITH_QUESTION)

    else:
        print('Plain BERT selected')

        run_multiple_epochs(N_EPOCH=N_EPOCH, LEARNING_RATE=LEARNING_RATE, BERT_DIR=BERT_DIR, INPUT_DIR=BODY_BODY_INPUT_DIR, \
            OUTPUT_DIR=RUN_DIR, BERT_CONFIG_DIR=BERT_CONFIG_DIR, MAX_SEQUENCE_LENGTH=384)
        
    
    #delte checkpoint not used no more
    delte_checkpoints(RUN_DIR)