# DeepSleepNet_Implementation_in_Keras

original paper : https://arxiv.org/abs/1703.04046

original implementation by the authors : https://github.com/akaraspt/deepsleepnet

I followed implementation of the deepsleepnet in tensorflow into keras (actually keras in tensorflow).

## Requirements

    conda create -n deepsleepnet_keras environment --file requirements.txt


## Data Preparation

Download and getting NPZ file is totally same with original implementation from https://github.com/akaraspt/deepsleepnet .

To get 20 fold cross validation data:

    python data_preparation.py
    
## Training

    python trainer.py
    
The trained model (both featurenet and deepsleepnet) will be stored at the **./weights** as weights.

## Performance Evaluation

    cat performance.txt
