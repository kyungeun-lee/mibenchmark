# MI estimation benchmark
The implementation codes of the paper "A Benchmark Suite for Evaluating Neural Mutual Information Estimators on Unstructured Datasets"
(Paper url: TBU)

## Table of Contents
- [Overview](#overview)
- [Required libraries](#required-libraries)
- [Dataset descriptions](#dataset-descriptions)
- [Examples](#examples)
- [License](#license)
- [Contact](#contact)

## Overview
We introduce a comprehensive benchmark suite for evaluating neural MI estimators on unstructured datasets, specifically focusing on images and texts. By leveraging same-class sampling for positive pairing and introducing a binary symmetric channel trick, we show that we can accurately manipulate true MI values of real-world datasets.

See our paper at [TBU].

## Required libraries
Our project has been built on Python 3.9. Here is the entire list of python libraries required (also available in `requirements.txt`):

``` swift
argparse==1.1
torch==2.0.1
torchvision==0.13.0a0
numpy==1.26.0
scipy==1.10.1
```

## Dataset descriptions
We propose a comprehensive method for evaluating neural MI estimators across various data domains. Specifically, this benchmark suite focuses on three types of data domains: (1) a multivariate Gaussian dataset, commonly used for evaluating MI estimators in existing works; (2) an image dataset consisting of digits, as an example of vision tasks; and (3) a sentence embedding dataset consisting of BERT embeddings of movie review datasets, as an example of NLP tasks. Our benchmark suite is built on publicly available datasets, such as MNIST and IMDB, which are freely available for academic and non-commercial use. For the IMDB datasets, we utilize the BERT fine-tuned sentence embeddings. The full list of datasets is available in the `dataset` directory.

##### 1. Multivariate Gaussian (`libs/utils_gaussian.py`) 
- The dataset is sampled from the multivariate Gaussian distribution in real-time.

##### 2. Images (`libs/utils_images.py`)
- In the current version, three types of image datasets are available: MNIST, CIFAR-10, and CIFAR-100.
- As described in the paper, our method allows the information source to be a binary random variable (to make use of bit-scale MI values). For instance, in the MNIST dataset, we use the classes 0 and 1, resulting in a total of 10,000 samples.
- To control the number of information sources, representation dimension, and nuisance strength, you can freely define `n_patch`, `img_size`, and `eta` in the code.

##### 3. Sentence embeddings (`libs/utils_text.py`)
- Similar to the image datasets, our method allows the information source to be a binary random variable (to make use of bit-scale MI values). We provide datasets in the `dataset` directory with classes 0 and 1.
- The dataset is sampled from a limited number of samples, with 12,500 samples in each class. Each sample is saved in `.npy` format.

##### 4. Mixture of images and sentence embeddings (`libs/utils_mixture.py`)
- To utilize the mixture of images and sentence embeddings, we sample `x` from images and `y` from sentence embeddings based on the predefined class information.


### Experiments
We provide the implementation code in `main.py`. Below are the descriptions for each argument:

- `gpu_id`: GPU index for training (dtype: `int`)
- `savepath`: Path to save the results (dtype: `str`)
- `ds`: number of information sources (dtype: `int`) -- The number of independent scalar random variables used to form the mutually shared information between `X` and `Y` (Definition 4.1)
- `dr`: representation dimension (dtype: `int`) -- The size of observational data (Definition 4.2)
- `dtype`: Data type (Options: gaussian, image, text, misture)
- `dname1`: Data name for images (Ignore this argument if you do not use image datasets.) (Options: mnist, cifar10, cifar100) 
- `dname2`: Data name for texts (Ignore this argument if you do not use text datasets.) (Options: imdb.bert-imdb-finetuned, imdb.roberta-imdb-finetuned)
- `nuisance`: Nuisance strength (Applicable only for image datasets.) (dtype: `float`)
- `output_scale`: MI value scales (Options: bit, nat)
- `critic_type`: Choices for the critic function $`f(x,y)`$ (See Section 2 for details.) (Options: inner, bilinear, separable, joint)
- `critic_depth`: Depth of the MLP critic (dtype: `int`)
- `critic_width`: Width of the MLP critic (dtype: `int`)
- `critic_embed`: Embedding size of the MLP critic (dtype: `int`)
- `estimator`: Types of neural MI estimators (Options: nwj, js, infonce, dv, mine, smile-1, smile-5, smile-inf)
- `gaussian_cubic`: For multivariate Gaussian datasets, set this to 1 to utilize $`y^3`$ instead of $`y`$. Otherwise, set it to 0. (dtype: `int`)
- `image_patches`: For image datasets, define the combination patterns as `[channel, width, height]` to combine multiple samples as described in Figure 2. (dtype: `str`)
- `image_channels`: Number of image channels for image datasets, i.e., RGB or grayscale. (dtype: `int`)
- `encoder`: Options for estimating MI between deep representations of images and texts (Options: None, irevnet, realnvp, maf, pretrained_resnet)
- `batch_size`: Batch size for training critics for MI estimation. (dtype: `int`)
- `learning_rate`: Learning rate training critics for MI estimation. (dtype: `float`)
- `n_steps`: Number of steps for training critics for MI estimation. (dtype: `int`)
- `mode`: Mode for setting true MI values. For `stepwise`, true MI values are set as [2, 4, 6, 8, 10] and change for `n_steps`//4. For `single`, true MI value is defined in the `true_mi` argument (Options: stepwise, single)
- `true_mi`: True MI value for estimation. This value is used to calculate the crossover probability of the binary symmetric channel. Ignore this argument if you set `mode` as `stepwise`. (dtype: `float`)

##### Default setup for training critic functions
- `critic_type`: joint
- `critic_depth`: 2
- `critic_width`: 256
- `critic_embed`: 32
- `batch_size`: 64
- `learning_rate`: 0.0005
- `n_steps`: 20000

### Examples
Here we provide the simple examples in step by step.
1. Clone the repository:
    ```sh
    git clone https://github.com/kyungeun.lee/mibenchmark.git
    ```
2. Navigate to the project directory:
    ```sh
    cd mibenchmark
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Estimate MI (You need to set the arguments based on the descriptions provided above.):
- An example of the multivariate Gaussians
    ```sh
    python main.py --gpu_id 0 --savepath results/gaussian --ds 10 --dr 10 --dtype gaussian --critic_type joint --estimator dv --mode stepwise
    ```
- An example of the images
    ```sh
    python main.py --gpu_id 0 --savepath results/images --ds 10 --dr 4096 --dtype image --critic_type joint --estimator dv --mode stepwise --dname1 mnist --image_patches "[1, 2, 5]" --image_channels 1
    ```
- An example of the sentence embeddings
    ```sh
    python main.py --gpu_id 0 --savepath results/texts --ds 10 --dr 7680 --dtype text --critic_type joint --estimator dv --mode stepwise --dname2 imdb.bert-imdb-finetuned
    ```
**RESULTS**
- Estimation logs into `mis.npy` file in the predetermined `savepath`.
- Estimation results used in the paper are available in `results/*`.
5. Analyze the estimation results: See the examples in `result_analysis.ipynb`.

### License
Apache-2.0 license

### Contact
Kyungeun Lee (e-mail: kyungeun.lee@lgresearch.ai)
