# Raga Training Repository

This repository handles the training for my science fair project, predicting ragas from an audio input file. The total project includes 5000+ lines of code across different branches and iterations of the system.

#### File Organization
|File|Description|
|---|---|
|[train.py](train.py)|Actually manages backpropagation, network states, and access data loaders. Reports all training and validation accuracies to a TenssorBoard data logger.|
|[data.py](data.py)|Handles data loading during training.|
|[models.py](models.py)|Network model files, it's a class for models|

#### Branch Organization
|Branch|Description|
|---|---|
|[Master (Stage #1 Training)](https://github.com/sauhaardac/raga_training/)|Stage #1 training of simple convolutional network on chunks of data|
|[Recurrent](https://github.com/sauhaardac/raga_training/tree/recurrent)|Conducts Stage #2 of training with additional two-layer LSTM network.| 
|[Leave One Validation](https://github.com/sauhaardac/raga_training/tree/leaveoneval)|Conducts thorough leave-one validation benchmark comparison for full recurrent PhonoNet architecture| 
## Contact Me
If you have any questions or need assistance with this repository in any way, do not hesitate to contact me at [sauhaarda@gmail.com](mailto:sauhaarda@gmail.com). Personal website available [here](sauhaarda.me) as well.
