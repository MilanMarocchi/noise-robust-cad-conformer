# Noise Robust CAD Conformer Library 


## Setup for development

Install the dependencies for the project

### Using virtual environements
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Using pipenv 
```
pipenv install
```

## Running

There are various files and script for various tasks, each frontend file will have a description of what it does and its arguments if you call it as follows.

```
python <script_name>.py --help
```

- You will first need to provide a csv of your data and create segmentation files using the matlab script:
- Next you will need to create a crossfold split file using this reference csv as follows:
- Now you can train a model using the run_model.py script and the train_audio_model command or run the paper trials such as tuning the mfcc conformer or 

### Running paper trials

To run the tuning run script `run_th_tine_mfcconformer.sh`

To run the baseline script for ten trials on the same split run `get_baseline.sh`