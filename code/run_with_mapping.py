import pandas as pd
import time, os, subprocess, argparse
import tensorflow as tf
from sklearn.metrics import accuracy_score

'''
This file runs the overlapping_question_2 dataset

it basically learns from question and argument if the argument argues for pro or con
'''
def run_with_mapping(N_EPOCH, LEARNING_RATE, BERT_DIR, BODY_BODY_INPUT_DIR, QUESTION_BODY_INPUT_DIR, OUTPUT_DIR, BERT_CONFIG_DIR, MAX_SEQUENCE_LENGTH=384, WITH_QUESTION=True):

    #defaut task
    if WITH_QUESTION:
        task_name = 'FNC1'
    else:
        task_name = 'FNC1_Body'

    #get body body ds and save a version with question body so bert can predict in OUTPUT_DIR
    df_train = pd.read_csv(BODY_BODY_INPUT_DIR+'train.tsv', sep='\t')
    df_val = pd.read_csv(BODY_BODY_INPUT_DIR+'val.tsv', sep='\t')

    def process_df(df):
        '''
        splits up body body rows into two samples with question-body
        '''
        #repeat rows
        df = pd.concat((df,df)).sort_index()
        #create index
        df = df.reset_index(drop=True).reset_index()
        df['index'] = df['index'].apply(lambda idx: (idx%2))
        df['id'] = df['id'].map(lambda idx: str(idx%2)+'_')+df['index'].map(str)

        #based on index save argument in body2 (where index = 0 copy body1 to body2)
        df['body2'][df['index']==0] = df['body1'][df['index']==0]

        #save question as body1
        df['body1']=df['question']
        del df['question']
        del df['index']

        return df

    #shift dataset
    df_val = process_df(df_val)
    df_train = process_df(df_train)

    print('df val: ',df_val.shape[0])
    print('df train: ',df_train.shape[0])

    #save new sets
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    df_train.to_csv(OUTPUT_DIR+'train.tsv',index=False, sep='\t')
    df_val.to_csv(OUTPUT_DIR+'val.tsv',index=False, sep='\t')

    #setting checkpoint
    checkpoint = BERT_DIR + 'bert_model.ckpt'
    #checkpoint = tf.train.latest_checkpoint(checkpoint_dir=OUTPUT_DIR + f'/epoch{4}')

    for epoch in range(0,N_EPOCH):

        #creating folder
        cur_epoch_dir = OUTPUT_DIR + f'epoch{epoch+1}'
        os.makedirs(cur_epoch_dir,exist_ok=True)
        print(f"CURRENT EPOCH DIRECTORY: {cur_epoch_dir}")

        bert_out = subprocess.check_output(["python", "./code/bert/run_classifier.py", f"--task_name={task_name}",
                                            "--do_train=true",
                                            "--do_eval=true",
                                            "--do_eval_train=true",
                                            "--do_predict=true",
                                            "--do_predict_train=true",
											'--do_lower_case=False',
                                            f"--data_dir={QUESTION_BODY_INPUT_DIR}",
                                            f"--vocab_file={BERT_DIR}vocab.txt",
                                            f"--bert_config_file={BERT_CONFIG_DIR}",
                                            f"--init_checkpoint={checkpoint}",
                                            f"--max_seq_length={MAX_SEQUENCE_LENGTH}",
                                            f"--train_batch_size={6}",
                                            f"--learning_rate={LEARNING_RATE}",
                                            f"--num_train_epochs={1}",
                                            f"--output_dir={cur_epoch_dir}"])

        checkpoint = tf.train.latest_checkpoint(checkpoint_dir=cur_epoch_dir)
        print(f'checkpint for next epoch: {checkpoint}')


        #predicting for body body dataset
        #first copy prediction qestion body
        os.rename(cur_epoch_dir+'/test_results.tsv', cur_epoch_dir+'/test_results_question_body.tsv')
        os.rename(cur_epoch_dir+'/train_results.tsv', cur_epoch_dir+'/train_results_question_body.tsv')

        os.rename(cur_epoch_dir+'/eval_results.txt', cur_epoch_dir+'/eval_result_question_body.txt')
        os.rename(cur_epoch_dir+'/eval_train_results.txt', cur_epoch_dir+'/eval_train_result_question_body.txt')

        #delete record files to force the model using the new ones
        os.remove(cur_epoch_dir+'/eval.tf_record')
        os.remove(cur_epoch_dir+'/train.tf_record')
        

        #because GPU is sometimes not freed fast enough sleep 5m
        time.sleep(300)

        #make bert prediction
        #eval not neccessary, because 
        bert_out = subprocess.check_output(["python", "./code/bert/run_classifier.py", f"--task_name={task_name}",
                                            "--do_train=false",
                                            "--do_eval=false",
                                            "--do_eval_train=fales",
                                            "--do_predict=true",
                                            "--do_predict_train=true",
											'--do_lower_case=False',
                                            f"--data_dir={OUTPUT_DIR}",
                                            f"--vocab_file={BERT_DIR}vocab.txt",
                                            f"--bert_config_file={BERT_CONFIG_DIR}",
                                            f"--init_checkpoint={checkpoint}",
                                            f"--max_seq_length={MAX_SEQUENCE_LENGTH}",
                                            f"--train_batch_size={6}",
                                            f"--learning_rate={LEARNING_RATE}",
                                            f"--num_train_epochs={1}",
                                            f"--output_dir={cur_epoch_dir}"])

        #because GPU is sometimes not freed fast enough sleep 5m
        time.sleep(300)

        #take prediction and base body body prediction on
        df_train_pred = pd.read_csv(cur_epoch_dir+'/train_results.tsv',sep='\t',header=None)
        df_val_pred = pd.read_csv(cur_epoch_dir+'/test_results.tsv',sep='\t',header=None)
        df_train_pred.columns=['agree','disagree']
        df_val_pred.columns=['agree','disagree']

        def map_results(df_pred):
            pred = df_pred.values
            
            pred = pred.reshape(-1,4)
            df_pred = pd.DataFrame(pred,columns=['agree1','disagree1','agree2','disagree2'])
            
            #mapping values to agree and disagree
            df_pred['agree'] = 2*(df_pred['agree1']-0.5)*(df_pred['agree2']-0.5)+0.5
            df_pred['disagree'] = 1.-df_pred['agree']

            return df_pred[['agree','disagree']]
        
        print(f'df_val_pred.shape: {df_val_pred.shape[0]}')
        print(f'df_train_pred.shape: {df_train_pred.shape[0]}')

        df_val = map_results(df_val_pred)
        df_train = map_results(df_train_pred)

        df_val.to_csv(cur_epoch_dir+'/test_results_processed.tsv',header=False, index=False, sep='\t')
        df_train.to_csv(cur_epoch_dir+'/train_results_processed.tsv',header=False, index=False, sep='\t')

        
