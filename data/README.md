# Preparation of dataset for each task
We use datasets as follows for each task. Please download data and place it in specified directory respectively.

<!-- |                     | ASV           | ER           | ASR             | IC                    | 
| ------------------- | ------------- | ------------ | --------------- | --------------------- | 
| Dataset | VoxCeleb | IEMOCAP | Librilight 10h  | Fluent Speech Command| 
| Link | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html      | https://sail.usc.edu/iemocap/index.html | https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz http://www.openslr.org/resources/12/dev-clean.tar.gz |  https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus |
| Directory | voxceleb | iemocap | librilight |fsc |    -->

## ASV
We use the VoxCeleb1 dataset for evaluation. The dev subset which
consists of 148,642 utterances from 1,251 speakers, approximately
351 hours in total is used for training. The cleaned
original test set which consists of 37,611 trials over 40 speakers
is used for testing.

You can download the datasets for training and testing at https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html.

Please place them in `'./data/voxceleb/'`.

## ER
We use the IEMOCAP dataset for evaluation.
It has approximately 12 hours of audio data with scripted and
improvised dialogues by 10 speakers. We performed the five
folds cross-validation, leaving the emotion labels “neutral”,
“happy”, “sad”, and “angry”. Note that, following the previous
work, “excite” is merged into “happy”.

You can download the dataset at  https://sail.usc.edu/iemocap/index.html. 

Please place it in `'./data/iemocap/'`.

## ASR
The LibriLight dataset (10 hour supervised subset) is used for training, which is a collection of spoken English audio derived from open-source audio books. The standard LibriSpeech dev set is used for testing.

You can download the training set at https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz and the dev set at http://www.openslr.org/resources/12/dev-clean.tar.gz . 

Please place them in `'./data/libright/'`.

## IC
We use the Fluent Speech Commands for evaluation. Fluent Speech Commands is a dataset of 30,043 English audios with 77 speakers, approximately 19 hours in total, each labeled with “action”, “object”, and “location” slots. The train subset is used for training and the test subset is used for testing.

You can download the datasets for training and testing at  https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus. 


Please place them in `'./data/fsc/'`.