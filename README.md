# FORWARD REACTION PREDICTION MODEL

This repo is based on the ["Biocatalysed synthesis planning using data-driven learning"](https://www.nature.com/articles/s41467-022-28536-w) publication. The following steps are for implementing the forward reaction prediction model.

## Requirements

The specific versions used in this project were:
Python: 3.6.9
ONMT Version: 1.0.0
RDKit: 2021.09.04

### Step 1: Conda Environment Setup

Because not all dependencies are available from PyPI for all platforms, we recommend creating a conda environment using the supplied conda.yml:

```bash
conda env create -f conda.yml
conda activate rxn-biocatalysis-tools
```

Alternatively, RXN Biocatalysis Tools can be obtained as a PyPI package and installed using pip. However, the installation may not include all dependencies, depending on your platform.

```bash
pip install rxn-biocatalysis-tools
pip install -r requirements.txt
pip install -r test_requirements.txt
pip install -r requirements_conda.txt
pip install OpenNMT-py==1.0.0
```

### Step 2: Data Preparation

The RXN Biocatalysis Tools Python package includes a script that can be utilized for preprocessing reaction data. Depending on the set options, the script generates an output data structure.

**[Download ECREACT 1.0](data/ecreact-nofilter-1.0.csv)**

```bash
python3 bin/rbt-preprocess.py data/ecreact-nofilter-1.0-copy.csv output --remove-patterns data/patterns.txt --remove-molecules data/molecules.txt --ec-level 3 --max-products 1 --min-atom-count 4 --bi-directional --split-products
```

### Step 3: Preprocess the data

The script below makes use of the data prepared in step 2. You may need to modify the paths depending on your directory structure and operating system.

```bash
DATASET=data/uspto_dataset
DATASET_TRANSFER=experiments/3

onmt_preprocess -train_src ${DATASET}/src-train.txt ${DATASET_TRANSFER}/src-train.txt -train_tgt ${DATASET}/tgt-train.txt ${DATASET_TRANSFER}/tgt-train.txt -train_ids uspto transfer  -valid_src ${DATASET_TRANSFER}/src-valid.txt -valid_tgt ${DATASET_TRANSFER}/tgt-valid.txt -save_data preprocessing/multitask_forward -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab --num_threads 4

```

The pre-processed USPTO files can be found [here](https://github.com/rxn4chemistry/OpenNMT-py/tree/carbohydrate_transformer/data/uspto_dataset). The data contains parallel precursors (`src`) and products (`tgt`) with one reaction per line and tokens separated by a space.

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

After running the preprocessing script, the following files are generated:

* `uspto.train.pt`: serialized PyTorch file containing training data
* `uspto.valid.pt`: serialized PyTorch file containing validation data
* `uspto.vocab.pt`: serialized PyTorch file containing vocabulary data

### Step 4: Train the model

The forward prediction model was trained using the following hyperparameters without a GPU.

#### Training the forward model

```bash
DATASET=preprocessing/multitask_forward
OUTDIR=model/multitask_forward
LOGDIR=logs/forward

WEIGHT1=9
WEIGHT2=1

onmt_train -data ${DATASET} \
    -save_model ${OUTDIR} \
    -data_ids uspto transfer --data_weights $WEIGHT1 $WEIGHT2 \
    -train_steps 250000 -param_init 0 \
    -param_init_glorot -max_generator_batches 32 \
    -batch_size 6144 -batch_type tokens \
    -normalization tokens -max_grad_norm 0  -accum_count 4 \
    -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam  \
    -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 \
    -layers 4 -rnn_size  384 -word_vec_size 384 \
    -encoder_type transformer -decoder_type transformer \
    -dropout 0.1 -position_encoding -share_embeddings  \
    -global_attention general -global_attention_function softmax \
    -self_attn_type scaled-dot -heads 8 -transformer_ff 2048 \
    --tensorboard --tensorboard_log_dir ${LOGDIR}
```

#### Continue training from the latest saved model

If training is interrupted at any point, it can be resumed from the most recent saved model by running the following script:

```bash
DATASET=preprocessing/multitask_forward
OUTDIR=model/multitask_forward
LOGDIR=logs/forward
MODEL=$(ls model/multitask_forward*.pt -t | head -1)


WEIGHT1=9
WEIGHT2=1

onmt_train -data ${DATASET} \
    -save_model ${OUTDIR} \
    -data_ids uspto transfer --data_weights $WEIGHT1 $WEIGHT2 \
    -train_steps 250000 -param_init 0 \
    -param_init_glorot -max_generator_batches 32 \
    -batch_size 6144 -batch_type tokens \
    -normalization tokens -max_grad_norm 0  -accum_count 4 \
    -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam  \
    -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 \
    -layers 4 -rnn_size  384 -word_vec_size 384 \
    -encoder_type transformer -decoder_type transformer \
    -dropout 0.1 -position_encoding -share_embeddings  \
    -global_attention general -global_attention_function softmax \
    -self_attn_type scaled-dot -heads 8 -transformer_ff 2048 \
    --train_from ${MODEL} \
    --tensorboard --tensorboard_log_dir ${LOGDIR}

```

### Step 5: Forward reaction prediction

Create a text file named `rxn.txt` that contains the reaction for which the product is to be predicted. Its format should be  `Reactant|EC#` (Example:`CC(N)=O.O|3.5.1.4`) Take note that the model interprets each line as a different reaction. So, several reactions could be predicted in a single text file. After running the following script, a file called `product.txt` will be created. The contents would include the predicted products of the forward reaction.

```bash
# forward prediction

OUTPUT_FOLDER="output/predictions"

# Obtain the most recent file from the model directory.
MODEL=$(ls model/multitask_forward*.pt -t | head -1)

LOGDIR=logs/forward

onmt_translate -model "${MODEL}" \
        -src "${OUTPUT_FOLDER}/rxn.txt" \
        -output "${OUTPUT_FOLDER}/product.txt" \
        -n_best 10 -beam_size 10 -max_length 300 -batch_size 64 --log_file_level  “${LOGDIR}”
```

## Notes

- Errors can occur when the dependencies are not compatible with your operating system. If that is the case, please read the error message and proceed to install the necessary version of the dependency.
- The data prepared and trained for the forward reaction prediction model by the author can be found in the `output` folder.