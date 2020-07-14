# Discriptions
This reporsitory is for rhythm-BERT, which is motivated by 'music is a kind of language'.

## The primitive original research proposal on May 31, 2020
- Colab link: https://drive.google.com/file/d/1PVrc_6QE-XM7rIpDUvYwPoOOAYVkOvtW/view?usp=sharing 
- Link of MIDI baselines mentioned in research proposals: https://drive.google.com/drive/folders/13eD6yUhjdHz639thV6LQQaKxKpVOZdVR?usp=sharing

Paper pending...

## File Organizations
Folder **colab_pruned** is a medley of experiments.
- **colab_pruned\data_folder** saves preprocessed data, including rhythm and chord dictionaries, Nottingham Dataset data in our formats. (Feel free to use pickle to read them, and you will get lists)
- **colab_pruned\model_VQ_EC2_BERT** is the model in my paper Word Representation for Rhythms. Feel free to run the updated notebook! (Remember to change paths!)
- **colab_pruned\Nottingham_database_preprocessing.ipynb** and **colab_pruned\data_loader_lib_trial.ipynb** are demos for using my data loader libs. You can get data in **colab_pruned\data_folder** by running those two files.
- **colab_pruned\model_seq2seq_baseline**, **colab_pruned\model_seq2seq_attention**, **colab_pruned\model_transformer** and **colab_pruned\pitch_encoder_decoder** are my primitive trials. I just save them here for reference.

Folder **lib**:
- **lib\data_loader** is for Nottingham Dataset preprocessing. It works, but it is still under maintain, because there are still a few bugs which do not matter greatly.
- **lib\models** and **lib\utils** are segments of BERT notebook. Just for reading. Of course, you can import them for further use.

# rhythm-word-embedding commit logs
This is for logging the research progressions of rhythm word embedding

## commits on June 28, 2020
- folder models: pitch encoder-decoder lib
- folder data_loader: Nottingham dataset loaders. How to use it? See /colab_pruned/data_loader_lib_trial.ipynb
- folder model_seq2seq_baseline and model_seq2seq_attention: RNN-autoencoder models for rhythm reconstruction
- folder model_transformer: transformer model for rhythm reconstruction and generation

## commits on July 7, 2020
- Folder orgnization changed! **lib** is a medley of modules, configs and toolkits; **colab_pruned** is a medley of previous experiments and implementations of lib.
- If you want to know rhythm words, please refer to '\colab_pruned\data_loader_lib_trial.ipynb' and '\colab_pruned\Nottingham_database_preprocessing.ipynb'
- If you want to know rhythm BERT, please refer to '\colab_pruned\model_VQ_EC2_BERT'
- If you want to implement codes yourself, please remember to change paths and configs yourself!

## commits on July 14, 2020
- Added fine-tuned code in this notebook: \colab_pruned\model_VQ_EC2_BERT\rhythm_VQ_EC2_BERT_released_20200714.ipynb
- Added midi reconstruction files
- Changed corresponding library

