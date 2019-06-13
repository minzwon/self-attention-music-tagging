## Toward Interpretable Music Tagging with Self-Attention
[Minz Won](https://minzwon.github.io/), [Sanghyuk Chun](https://sanghyukchun.github.io/home/), and [Xavier Serra](https://www.upf.edu/web/xavier-serra)

- Full paper (6 pages) [[ArXiv](https://arxiv.org/abs/1906.04972)]

- Machine Learning for Music Discovery Workshop at ICML 2019 (2 pages) [[Link](https://drive.google.com/file/d/1mYU1fjXkrcQBpTyzuCszyceBm2yNC9_O/view)]

### Abstract
Self-attention is an attention mechanism that learns a representation by relating different positions in the sequence. The transformer, which is a sequence model solely based on self-attention, and its variants achieved state-of-the-art results in many natural language processing tasks. Since music composes its semantics based on the relations between components in sparse positions, adopting the self-attention mechanism to solve music information retrieval (MIR) problems can be beneficial. 

Hence, we propose a self-attention based deep sequence model for music tagging. The proposed architecture consists of shallow convolutional layers followed by stacked Transformer encoders. Compared to conventional approaches using fully convolutional or recurrent neural networks, our model is more interpretable while reporting competitive results. We validate the performance of our model with the MagnaTagATune and the Million Song Dataset. In addition, we demonstrate the interpretability of the proposed architecture with a heat map visualization.

### Environment
This code is based on Python 3. Used libraries can be installed through pip.

`pip install -r requirements.txt`


### Preprocessing
1. Data split

	`python preprocessing/mtat/split.py run your_data_path`
	
2. Get log mel-spectrograms

	`python preprocessing/mtat/preprocess.py run your_audio_path your_data_path`
	
	
### Training
```
python main.py --architecture 'pons_won' --conv_channels 16 \
--attention_channels 512 --attention_layers 2 --attention_heads 8 \
--batch_size 16 \
--dataset 'mtat' --data_type 'spec' \
--data_path your_data_path --model_save_path your_model_path
```

### Evaluation
```
python eval.py --architecture 'pons_won' --conv_channels 16 \
--attention_channels 512 --attention_layers 2 --attention_heads 8 \
--dataset 'mtat' --data_type 'spec' \
--data_path your_data_path --model_load_path your_model_path
```

### Pre-trained model
Use `models/ponswon.pth`.

### Visualization
Check `visualize/visualize_example.ipynb`.

### Citation
```
@article{won2019attention,
    title={Toward Interpretable Music Tagging with Self-Attention},
    author = {Won, Minz and Chun, Sanghyuk and Serra, Xavier},
    journal={arXiv preprint arXiv:1906.04972},
    year={2019}
}
```
