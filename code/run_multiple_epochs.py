import pandas as pd
import time
import os
import subprocess
import argparse
import tensorflow as tf


def run_multiple_epochs(N_EPOCH, LEARNING_RATE, BERT_DIR, INPUT_DIR, OUTPUT_DIR, BERT_CONFIG_DIR, MAX_SEQUENCE_LENGTH=384):

    print(f"INPUT_DIR: {INPUT_DIR}")

	#setting checkpoint
    checkpoint = BERT_DIR + 'bert_model.ckpt'
    #checkpoint = tf.train.latest_checkpoint(checkpoint_dir=OUTPUT_DIR + f'/epoch{4}')

    for epoch in range(0, N_EPOCH):

        # creating folder
        cur_epoch_dir = OUTPUT_DIR + f'epoch{epoch+1}'
        os.makedirs(cur_epoch_dir, exist_ok=True)
        print(f"CURRENT EPOCH DIRECTORY: {cur_epoch_dir}")

        #run BERT
        bert_out = subprocess.check_output(["python", "./code/bert/run_classifier.py", "--task_name=FNC1",
                                            "--do_train=true",
                                            "--do_eval=true",
                                            "--do_eval_train=true",
                                            "--do_predict=true",
                                            "--do_predict_train=true",
											'--do_lower_case=False',
                                            f"--data_dir={INPUT_DIR}",
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
