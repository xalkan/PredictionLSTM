
  

# PredictionLSTM

Using a [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) to predict Google stock.
 

<img  src="https://github.com/xalkan/PredictionLSTM/blob/master/output.gif" />


Repo contains a h5py file with model architecture and weights trained on 100 epochs.
If you want to train your own model, then delete 'google_prediction_lstm.h5' file.

### Build and Run

Clone the repo and fire up a terminal in current working directory.

Create a virutual environment

  

    python -m venv venv

Activate the virtual environment

  

    # On Linux
    
    venv/bin/activate
    
    # On Windows
    
    cd venv/scripts/
    source activate

Install all the dependencies in requirements.txt

  

    pip install -r requirements.txt

  

Run the neural network

  

    python rnn.py
