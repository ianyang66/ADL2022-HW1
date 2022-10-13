# ADL2022-HW1

## Objective
Implement LSTM-based and GRU-based method to achieve intent classification and slot tagging.  
For more details, please view my report https://www.dropbox.com/s/55vsj8k60jdbwqj/report.pdf?dl=1.


## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
# otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```

## Intent detection
### train
```shell
python train_intent.py --data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device> --seed <seed>
```
* **data_dir**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_dir**: Directory to save the model file.
* **max_len**: Number of max length.
* **hidden_size**: RNN hidden state dim.
* **num_layers**: Number of layers.
* **dropout**: Model dropout rate.
* **bidirectional**: Whether the model is bidirectional
* **lr**: Optimizer learing rate.
* **batch_size**: Number of batch size.
* **device**: Choose your device (cuda, cpu, cuda:0 or ...).
* **seed**: Number of seed. It make prediction can reproduce.

### output csv
```shell
python test_intent.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device>
```
* **test_file**: Path to the test file.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_path**: Path to model checkpoint.
* **pred_file**: Perdict file path.
* **max_len**: Number of max length.
* **hidden_size**: RNN hidden state dim.
* **num_layers**: Number of layers.
* **dropout**: Model dropout rate.
* **bidirectional**: Whether the model is bidirectional
* **batch_size**: Number of batch size.
* **device**: Choose your device (cuda, cpu, cuda:0 or ...).


### reproduce my result (Public on Kaggle: 0.91466)
```shell
bash download.sh
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

---

## Slot tagging
### train
```shell
python train_slot.py --data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device> --seed <seed>
```
* **data_dir**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_dir**: Directory to save the model file.
* **max_len**: Number of max length.
* **hidden_size**: RNN hidden state dim.
* **num_layers**: Number of layers.
* **dropout**: Model dropout rate.
* **bidirectional**: Whether the model is bidirectional
* **lr**: Optimizer learing rate.
* **batch_size**: Number of batch size.
* **device**: Choose your device (cuda, cpu, cuda:0 or ...).
* **seed**: Number of seed. It make prediction can reproduce.

### output csv
```shell
python test_slot.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device>
```
* **test_file**: Path to the test file.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_path**: Path to model checkpoint.
* **pred_file**: Perdict file path.
* **max_len**: Number of max length.
* **hidden_size**: RNN hidden state dim.
* **num_layers**: Number of layers.
* **dropout**: Model dropout rate.
* **bidirectional**: Whether the model is bidirectional
* **batch_size**: Number of batch size.
* **device**: Choose your device (cuda, cpu, cuda:0 or ...).


### reproduce my result (Public on Kaggle: 0.82252)
```shell
bash download.sh
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```

## Reference
[ntu-adl-ta/ADL21-HW1](https://github.com/ntu-adl-ta/ADL21-HW1)
