# rhythm-word-embedding
This is for logging the research progressions of rhythm word embedding

## research proposal on May 31, 2020
- Colab link: https://drive.google.com/file/d/1PVrc_6QE-XM7rIpDUvYwPoOOAYVkOvtW/view?usp=sharing 
- Link of MIDI baselines mentioned in research proposals: https://drive.google.com/drive/folders/13eD6yUhjdHz639thV6LQQaKxKpVOZdVR?usp=sharing

## commits on June 28, 2020
- folder models: pitch encoder-decoder lib
- folder data_loader: Nottingham dataset loaders. How to use it? See /colab_pruned/data_loader/data_loader_lib_trial.ipynb
- folder model_seq2seq_baseline and model_seq2seq_attention: RNN-autoencoder models for rhythm reconstruction
- folder model_transformer: transformer model for rhythm reconstruction and generation

## commits on July 7, 2020
- Folder orgnization changed! **lib** is a medley of modules, configs and toolkits; **colab_pruned** is a medley of previous experiments and implementations of lib.
- If you want to know rhythm words, please refer to '\colab_pruned\data_loader_lib_trial.ipynb' and '\colab_pruned\Nottingham_database_preprocessing.ipynb'
- If you want to know rhythm BERT, please refer to '\colab_pruned\model_VQ_EC2_BERT'
- If you want to implement codes yourself, please remember to change paths and configs yourself!

