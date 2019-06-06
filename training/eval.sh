python -u eval.py architecture 'pons_won' \
  --conv_channels 16 --batch_size 16 \
  --dataset 'mtat' --data_type 'spec' \
  --audio_path '/datasets/MTG/users/mwon/mtat' --model_path './../models/mtat_pons_won/5_G.pth'
