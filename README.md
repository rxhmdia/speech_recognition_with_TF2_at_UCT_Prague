# Character-level DNN Model for Czech Language Speech Recognition using TensorFlow 2

A project which was given birth as a masters thesis at UCT Prague and now continues as a part of my PhD studies. 
The core concept is developing a Character-level deep neural network (DNN) based model for 
automatic speech recognition (ASR) and transcription of Czech naturally spoken language. 
Ultimately the transcription is then to be used for controlling a robotic system using natural Czech spoken language, 
utilizing keyword spotting mechanisms.

The system starts with a data transformation pipeline, which feeds
preprocessed speech data from two Czech natural speech corpuses ([PDTSC 1.0](https://ufal.mff.cuni.cz/pdtsc1.0/en/index.html) and [ORAL2013](https://wiki.korpus.cz/doku.php/en:cnk:oral2013)) into an Acoustic Model (AM).
Data Augmentation techniques, such as [SpecAugment](https://arxiv.org/abs/1904.08779) are implemented into the pipeline for improving performance on testing dataset.

## Table of contents
* [Getting Started](#getting-started)
    * [Requirements](#requirements)
    * [Preparing Datasets for Training](#preparing-datasets-for-training)
    * [Training](#training)
    * [Production](#production)
* [Project Status](#project-status)
    * [Implemented Functionality](#implemented-functionality)
    * [Currently Working on](#currently-working-on)
    * [Future Plans](#future-plans)
* [Versioning](#versioning)
* [Authors](#authors)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

The environment was created and the required packaged were installed using the [Anaconda](https://www.anaconda.com) 
Python library management platform which can be downloaded from [Anaconda Distribution](https://www.anaconda.com/distribution/).

You can recreate the environment with all the required packages using the 
[environment.yml](https://github.com/vejvarm/speech_recognition_with_TF2_at_UCT_Prague/blob/master/environment.yml) 
file in this repository by following instructions from [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Once conda is installed, all you have to do is open Anaconda Prompt and input:
```
conda env create -f environment.yml
```

### Preparing Datasets for Training
In order to train the network, you need to have a speech dataset transcripts in czech language. 
In this project, training was done on aforementioned [PDTSC 1.0](https://ufal.mff.cuni.cz/pdtsc1.0/en/index.html)
and [ORAL2013](https://wiki.korpus.cz/doku.php/en:cnk:oral2013) datasets. 
For these to work, they need to be preprocessed and transformed into MFCC/MFSC feature structures of correct shape 
and encoded into .tfrecord format, which is done by calling the `DataPrep` class from `DataOps.py`. 

Example for calling `DataPrep` on raw PDTSC dataset files with default preprocessing settings:

```
audio_folder = "path/to/PDTSC/folder/raw/audio/"
transcript_folder = "path/to/PDTSC/folder/raw/transcripts/"
save_folder = 'path/to/output/folder'

dp = DataPrep(audio_folder, transcript_folder, save_folder)

dp.run()
```
<details>
<summary><strong>Mandatory arguments in <code>DataPrep</code> class:</strong></summary>

 - `audio_folder` _(string)_: path to folder with raw audio files (.wav or .ogg)
 - `transcript_folder` _(string)_: path to folder with raw transcript files (.txt)
 - `save_folder` _(string)_: path to folder in which to save the preprocessed data
</details>

<details>
<summary><strong>Optional keyword arguments in <code>DataPrep</code> class:</strong></summary>

 - `dataset` _(string)_: which dataset is to be expected (allowed:"pdtsc" or "oral")
 - `feature_type` _(string)_: which feature type should the data be converted to (allowed: "MFSC" or "MFCC")
 - `label_type` _(string)_: type of labels (so far only "unigram" is implemented)
 - `repeated` _(bool)_: whether the bigrams should contain repeated characters (eg: 'aa', 'bb')
 - `energy` _(bool)_: whether energy feature should be included into feature matrix
 - `deltas` _(Tuple[int, int])_: area from which to calculate differences for deltas and delta-deltas
 - `nbanks` _(int)_: number of mel-scaled filter banks
 - `filter_nan` _(bool)_: whether to filter-out inputs with NaN values
 - `sort` _(bool)_: whether to sort resulting cepstra by file size (i.e. audio length)
 - `label_max_duration` _(float)_: maximum time duration of the audio utterances
 - `speeds` _(Tuple[float, ...])_: speed augmentation multipliers (between 0. and 1.)
 - `min_frame_length` _(int)_: signals with less time-frames will be excluded
 - `max_frame_length` _(int)_: signals with more time-frames will be excluded
 - `mode` _(string)_: whether to copy or move the not excluded files to a new folder
 - `delete_unused` _(bool)_: whether to delete files that were unused in the final dataset
 - `feature_names` _(string)_: part of filename that all feature files have in common 
 - `label_names` _(string)_: part of filename that all label files have in common
 - `tt_split_ratio` _(float)_: split ratio of training and testing data files (between 0. and 1.)
 - `train_shard_size` _(int)_: approximate tfrecord shard sizes for training data (in MB)
 - `test_shard_size` _(int)_: approximate tfrecord shard sizes for testing data (in MB)
 - `delete_converted` _(bool)_: whether to delete .npy shard folders that were already converted to .tfrecords
 - `debug` _(bool)_: switch between normal and debug mode
</details>

<details>
<summary><strong>Default/Allowed values of the keyword arguments in <code>DataPrep</code> class:</strong></summary>

```
__datasets = ("pdtsc", "oral")  # default choice: [0]
__feature_types = ("MFSC", "MFCC")  # default choice: [0]
__label_types = ("unigram", "bigram")  # default choice: [0]
__repeated = False
__energy = True
__deltas = (2, 2)
__nbanks = 40
__filter_nan = True
__sort = False
__label_max_duration = 10.0
__speeds = (1.0, )
__min_frame_length = 100
__max_frame_length = 3000
__modes = ('copy', 'move')  # default choice: [0]
__delete_unused = False
__feature_names = 'cepstrum'
__label_names = 'transcript'
__tt_split_ratio = 0.9
__train_shard_size = 2**10
__test_shard_size = 2**7
__delete_converted = False
__debug = False
```
</details>

### Training
Starting the training process itself is quite simple. Just run `main.py` with desired settings which
are determined by the [FLAGS.py](FLAGS.py) file. The following params can be changed and tweaked:

<details>
 <summary>Show list of all params.</summary>
 
 - `logger_level` _(string)_: verbosity level of the console logger ("DEBUG", "__INFO__", "WARNING")
 - `load_dir` _(string)_:  path to directory with the preprocessed data for training 
 - `save_dir` _(string)_: path to directory for saving model checkpoints and other data
 - `save_config_as` _(string)_: name of the config file backup in the save_dir (__"FLAGS.py"__)
 - `checkpoint_path` _(string)_: path to _model.h5_ checkpoint file for initialization from trained model
 - `num_runs` _(int)_: number of independent runs (repeats) of the entire training process (__5__)
 - `max_epochs` _(int)_: maximum number of epochs in each run (__20__)
 - `batch_size_per_GPU` _(int)_: size of training minibatches for each working GPU (__8__)
 - `shuffle_seed` _(int)_: seed for reproducing random shuffling order (__42__)
 - `bucket_width` _(int)_: size of buckets in which similar length utterances are grouped (__100__)
 - `save_architecture_image` _(bool)_: whether to save model architecture to `save_dir`
 - `show_shapes` _(bool)_: whether to also show model values for layer shapes in architecture image
 - `weight_init_mean` _(float)_: model weight random initialization mean value (__0.0__)
 - `weight_init_stddev` _(float)_: model weight random initialization standard deviation value (__0.0001__)
 - `ff_first_params` _(Dict)_: params for allowing/tweaking Dense layers at start of the model 
    - `use` _(bool)_: whether to include Dense layers at the start of the model or not (__False__)
    - `num_units` _(List[int])_: List/Tuple signifying number of layers and their number of hidden units
    - `batch_norm` _(bool)_: whether to include Batch Normalization layers after each Dense layer
    - `drop_rates` _(List[float])_: List/Tuple of dropout rates in each Dense layer
 - `conv_params` _(Dict)_: params for allowing/tweaking Convolutional layers at start of the model
    - `use` _(bool)_: wheter to include Convolutional layers at the start of the model or not (__True__)
    - `channels` _(List[int])_: List/Tuple for setting number of filters (channels) in each layer
    - `kernels` _(List[Tuple[int]])_: List/Tuple of Tuples (__time dim, feature dim__) for sizes of filters
    - `strides` _(List[Tuple[int]])_: List/Tuple of Tuples (__time dim, feature dim__) for strides of filters
    - `dilation_rates` _(List[Tuple[int]])_: List/Tuple of Tuples (__time dim, feature dim__) for filter dilations
    - `padding` _(str)_: whether/how to pad at the sides of the input 
        - ___'same'___ (default): padding at sides so that inp shape == out shape (not regarding strides)
        - __'valid'__: no padding at sides, meaning that output shape depends on kernel sizes
    - `data_format` _(str)_: order of dimensions in input data
        - __'channels_first'__: "NCHW" ... channels are in the first dimension (N is batch size)
        - ___'channels_last'___ (default): "NHWC" ... channels are in the last dimension (N is batch size)
    - `batch_norm` _(bool)_: whether to include Batch Normalization layers after each Conv layer
    - `drop_rates` (_List[float]_): List/Tuple of dropout rates in each Conv layer
 - `bn_momentum` _(float)_: momentum of Batch Normalization parameter updates (__0.9__)
 - `relu_clip_val` _(float)_: if the ReLU output will be higher than this number, it will get clipped (__20.__)
 - `relu_alpha` _(float)_: Leaky ReLU negative domain slope (__0.2__)
 - `rnn_params` _(Dict)_: params for allowing/tweaking Bidirectional Recurrent layers (BGRU) in the model
    - `use` _(bool)_: whether to include BGRU layers in the model (__True__)
    - `num_units` (_List[int]_): number of BGRU layers and number of their hidden neurons
    - `batch_norm` _(bool)_: whether to include Batch Normalization layers after the BGRU layers (__True__)
    - `drop_rates` _(List[float]): dropout rates in each BGRU layer
 - `ff_params` _(Dict)_: params for allowing Dense (FF) layers after the BGRU layers (__True__)
    - `use` _(bool)_: whether to include FF layers in the model (__True__)
    - `num_units` (_List[int]_): number of FF layers and number of hidden units (neurons) in each of them
    - `batch_norm` _(bool)_: whether to include Batch Normalization layers after the FF layers (__True__)
    - `drop_rates` _(List[float]): dropout rates in each FF layer
 - `lr` _(float)_: learning rate of the optimizer (__0.001__)
 - `lr_decay` _(bool)_: whether to exponentially decay learning rate during training (__True__)
 - `lr_decay_rate` _(float)_: rate at which the learning rate decays (__0.8__)
 - `lr_decay_epochs` _(int)_: how often will the learning rate be decayed (__1__)
 - `epsilon` _(int)_: Adam optimizer parameter to prevent division by zero (__0.1__)
 - `amsgrad` _(bool)_: Switch between Adam and AMSGrad optimizer (__True__ is for AMSGrad)
 - `data_aug` _(Dict)_: Pipeline Data Augmentation parameter dictionary
    - `mode` _(str)_: how many times to apply SpecAugment on input data ("0x", __"1x"__, "2x")
    - `bandwidth_time` (_Tuple[int]_): time masking bandwidth range (__(10, 100)__)
    - `bandwidth_freq` (_Tuple[int]_): frequency masking bandwidth range (__(10, 30)__)
    - `max_percent_time` _(float)_: maximum time percentage that can be masked (__0.2__)
    - `max_percent_freq` _(float)_: maximum frequency percentage that can be masekd (__1.0__)
 - `beam_width` _(int)_: Beam Search decoder beam width (__256__)
 - `top_paths` _(int)_: Number of best paths to be selected by Beam Search (__1__, >1 not supported)
 - `patience_epochs` _(int)_: Early Stopping patience before prematurely ending the trainig run (__3__)

</details>

### Production
__TODO__
```
class PREDICTION_FLAGS(FLAGS):

    recording = {
        "rate": 16000,
        "updates_per_second": 10,
        "channels": 1,
        "max_record_seconds": 30,
    }

    features = {
        "type": "MFSC",
        "energy": True,
        "deltas": (2, 2),
    }

    model = {
        "path": "path/to/trained/model.h5",
    }

    # Prediction
    beam_width = 256
    top_paths = 5
```

## Project Status

### Implemented Functionality
 - `FeatureExtraction.py` converting raw singals to MFCC or MFSC features
 - `DataOps.py` preprocessing and data preparation pipeline
   - `DataLoader` raw data loading
     - `PDTSCLoader` loader for raw [PDTSC 1.0](https://ufal.mff.cuni.cz/pdtsc1.0/en/index.html) dataset
     - `OralLoader` loader for raw [ORAL2013](https://wiki.korpus.cz/doku.php/en:cnk:oral2013) dataset
   - `DataPrep` automated preproc. pipeline for [PDTSC 1.0](https://ufal.mff.cuni.cz/pdtsc1.0/en/index.html) and [ORAL2013](https://wiki.korpus.cz/doku.php/en:cnk:oral2013) datasets
   - `SpecAug` data augmentation
 - `Model.py` character-level Acoustic Model (AM)
   - Convolutional, GRU recurrent and Feed-Forward layers
   - Connectionist Temporal Classification (CTC)
   - Learning rate decay
   - Early stopping
   - Batch Normalization
   
### Currently Working on
Currently the project is in the Data Augmentation implementation and testing stages.

### Future Plans
 - Better pipeline for DataPrep to reduce drive space requirements 
 
## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/vejvarm/speech_recognition_with_TF2_at_UCT_Prague/tags). 

## Authors

* **Martin Vejvar** - *Core developer* - [vejvarm](https://github.com/vejvarm)
* **Dovzhenko Nikita** - *Supporting developer*

## License

This project is licensed under the Academic Free License version 3.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* [How to write a good README for your GitHub project](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
* [A template to make good README.md](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2#file-readme-template-md)


