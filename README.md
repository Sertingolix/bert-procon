# BERT Model Bachelor thesis Lucas Brunner

<p align="center">
<img src="https://mtc.ethz.ch/_jcr_content/rightpar/contextinfo/fullwidthimage/image.imageformat.context.180972075.svg" alt="MTC ETHZ" width="25%"/>
</p>

## Usage Instructions

This repo cotnains the BERT model used in the Bachelor thesis of Lucas Brunner. Executing `bash startup.sh` installs Tensorflow GPU and other dependencies in a python enviroment. The system and software requirements for Tensorflow GPU can be found on `https://www.tensorflow.org/install/gpu`. `bash run.sh` executes BERT with default settings. If no pretrained BERT model is submitted (`--bert_dir path/to/BERT/`) the script will download the lates recomended BERT Base model (currently `multi_cased_L-12_H-768_A-12`). Similarly, if no dataset is submitted (`--input_dir path/to/dataset/folder`), the script will download/crawl the ProCon dataset (``). Those downloads are only executed once. Note that crawling the ProCon dataset takes ~5-10 minutes. A whole list of all suported settings can be accesed by executing `bash run.sh --h`.



### TL;DR
Install dependencies to run BERT in a python enviroment with:

```
bash startup.sh
```

Run BERT with default setting:

```
bash run.sh
```

To show all options use:

```
bash run.sh --h
```

## Credits

The BERT code is based on https://github.com/google-research/bert and adjusted to our requirements. The only file edited in the bert folder is run_classifier.py

