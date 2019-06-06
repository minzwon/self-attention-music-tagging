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

