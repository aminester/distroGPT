
# distroGPT: 

A simple and fast repository for training and fine-tuning medium-sized GPT models. Through the train.py file, this is capable of reproducing GPT-2 (124M) on OpenWebText using a single 8XA100 40GB node, with training taking about 4 days. The code is straightforward and easy to understand, with train.py being a acting as 300-line boilerplate training loop and model.py acting as a 300-line definition of the GPT model that can optionally load the GPT-2 weights from OpenAI.

<img width="603" alt="gpt2_124M_loss" src="https://user-images.githubusercontent.com/122713100/217517691-acb3c1a5-2ae6-4fca-8e94-acfae32ad1bd.png">

The code's simplicity makes it versatile for a number of use cases:

- Train new models from the ground up
- Finetune pretrained checkpoints (OpenAI's GPT-2 1.3B model is the biggest one available at the moment)


# Installation:

**Dependencies:**

- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install wandb` for optional logging <3
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install tqdm`
- [NumPy](https://numpy.org/install/)
- [PyTorch](https://pytorch.org)

# How to start:

A basic use case to start with is using Shakespeare to train the character-level GPT.  Start by downloading the text as a single 1MB file, then convert it from raw text into a single stream of integers."
```
$ python data/shakespeare_char/prepare.py
```

This action generates two files named train.bin and val.bin in the designated data directory. These files will be used to train your new GPT model. The model is scalable and thus the size of the model depends on the computing power of your system. 

If you have a Graphics Processing Unit (GPU), you can quickly train a smaller version of the GPT using the settings in the "config/train_shakespeare_char.py" configuration file.
```
$ python train.py config/train_shakespeare_char.py
```

This code trains a GPT model with a context size of up to 256 characters and 384 feature channels. It consists of a 6-layer Transformer with 6 heads in each layer. When run on a single A100 GPU, the training takes approximately 3 minutes, and the best validation loss achieved is 1.4697. The model checkpoints are stored in the "--out_dir" directory named "out-shakespeare-char." Upon completion of the training, the best model can be sampled by directing the sampling script to this directory.
```
$ python sample.py --out_dir=out-shakespeare-char
```

This outputs some few samples, such as the following examples below:

```
LUCIO:
And cowards shall be cast aside,
Their threats dismissed with pride,
For he who ran away, and hung
Had one with him, unsung.

ISABELLA:
I cannot countenance such a sight.

ISABELLA:
Then I must challenge him, to save the innocent:
And what have you, oppressor, to say for your defense?

ISABELLA:
If you have done wrongs of all descriptions,
To destroy the innocent's conviction,
The day of reckoning must come for all men,
That I will fight for, with heart and pen.
```

For this particular output, this was achieved after 2 minutes of training on an M1 MacBook Pro. This process can be optimized to output stronger results by calibrating a pretrained GPT-2 model with this dataset or making use of a more capable system.

For efficiency, we can use PyTorch nightly ([install](https://pytorch.org/get-started/locally/)) in place of the current protocol. However, we can still conduct a train run without nightly:

```
$ python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, considering we are using a CPU in place of a GPU, we must set `--device=cpu` and turn off PyTorch 2.0 compile with `--compile=False`. This results in a faster but noisier evaluation (`--eval_iters=20`, down from 200), a smaller context size of 64 characters in place of 256, and a smaller batch size of 12 examples per iteration in place of 64. Additionally, we'll use a smaller Transformer model with 4 layers, 4 heads, and a 128 embedding size, and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around to around max_iters with `--lr_decay_iters`). Given the small size of the network, we'll also reduce regularization (`--dropout=0.0`). Despite these changes, the training process still takes about 3 minutes and yields a 1.88 loss, which leads to worse generated samples:

```
PRINCE HAL:
Forth, yon thrum do hark to yond wild,
Where barks some knell thou chimed no in heatth
Forsooth, the weight's hence hent thou mett
And sought hap-so with glee shall moot ye wept.
```

You can also adjust the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, and any other necessary parameters. However, this will result in a noticeably longer wait time depending on how many parameters are changed.

For Apple Silicon MacBooks (e.g. M1, M2) with a recent PyTorch version, add the --device mps flag to utilize the on-chip Neural Engine. This optimized protocol significantly accelerates training (two to three-fold) and enables the use of larger networks.

## Recreating GPT-2

An application more focused on deep learning would be oriented around reproducing GPT-2 results. This application requires us to tokenize the dataset first. In this instance, we use [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```
$ python data/openwebtext/prepare.py
```

The process downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset, resulting in the creation of `train.bin` and `val.bin` containing GPT2 BPE token ids as raw uint16 bytes. To train a GPT-2 (124M) model, an 8X A100 40GB node is required, and the following command should be executed:

```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

The training process, using PyTorch Distributed Data Parallel (DDP), will take approximately 4 days and result in a loss of approximately 2.85. An evaluation of the GPT-2 model on the OpenWebText (OWT) dataset yields a validation loss of around 3.11. However, finetuning the model can reduce the loss to around 2.85, bridging the gap between the two models.

In a cluster environment with multiple GPU nodes, it is possible to harness the power of multiple GPUs, for example across 2 nodes:
```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is recommended to benchmark your interconnect using tools such as iperf3. If Infiniband is not available, add the `NCCL_IB_DISABLE=1` flag to the launches. Multinode training may still be possible, but performance may be significantly reduced without Infiniband. Checkpoints are automatically saved in the specified `--out_dir` during training. To sample from the trained model, run `python sample.py`.

Examine all the arguments of the script, as it aims to be readable, easily modifiable, and transparent. Depending on your requirements, you may need to adjust several of the variables.

# Baselines:

We can get some baselines in place for OWT using OpenAI GPT-2 checkpoints. We can get the numbers as follows:

```
$ python train.py eval_gpt2
$ python train.py eval_gpt2_medium
$ python train.py eval_gpt2_large
$ python train.py eval_gpt2_xl
```

and notice the train and val losses:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

We must consider that GPT-2 was trained on (closed-source) WebText, while OpenWebText is simply a faithful open-source reproduction of the dataset, resulting in a domain gap. When we taking the checkpoint for GPT-2 (124M) and adjust it on OpenWebText directly, it eventually arrives at a 2.85 loss, which can be interpreted as a more appropriate baseline reproduction.

# Finetuning:

Finetuning is a similar process to training, but we make take steps to initialize from an already pretrained model and use smaller learning rate in training. Finetuning is a relatively fast process, taking only a few minutes to complete using a single GPU. take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```
$ python train.py config/finetune_shakespeare.py
```

This process will load the config parameter overrides in `config/finetune_shakespeare.py`. We use a GPT2 checkpoint to initialize from using `init_from` and train, with the notable exception of using a small learning rate and shorter process. Finetuning may run up the system memory, in which case decreasing the model size is helpful (drawing from `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or just decreasing the `block_size'. The lowest validation loss will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. After this, you can run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

In this instance, the hyperparameters in the configuration were not tuned to a significant degree but doing so may yield better results.

# Sampling and Inference:

Sample from a pretrained model or a model you have configured yourself using the script `sample.py`. The following example provides a method to sample from the largest available `gpt2-xl` model:

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you are sampling from a model you have personally trained, point the code appropriately using `--out_dir`. It is also possible to use text from a file (e.g. `$ python sample.py --start=FILE:prompt.txt`) to prompt the model.

# Efficiency Notes:

  - Using `bench.py` is useful in cases for simple model profiling and benchmarking. It executes the same processes that occur in the `train.py` training loop while excluding any complexities that may hamper efficiency.

  -This code uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) by default. 
    - `torch.compile()` is available in the PyTorch nightly release (as of 02/08/2023). 
    -This brings a significant performance improvement, in some cases reducing iteration time from ~250ms / iter to 135ms / iter.

# Troubleshooting:

This code runs PyTorch 2.0 as a default, which may cause some compatibility issues on certain operating systems (notably Windows). 
  - Adding the `--compile=False` flag is possible fix for any linked error messages
  - This disables PyTorch 2.0 and makes the code runnable, but it throttles the speed as a tradeoff.
  
# License:

This project is licensed under the MIT License - see the LICENSE file for details.

