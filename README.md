# hml

## Usage Instructions

* You probably want to change `checkpoint_dir` and `result_dir` in the main scripts.

* Tags are used to specify new experiments and make new subfolders.

## Running Examples

* Run neural process model on Sinusoid dataset, with 2 GPUs and a meta batch size of 16
```
python run_1d_np.py --gpus 0,1 --nr_model 16 --learning_rate 0.0001 --dataset_name sinusoid
```

* Run MAML model on Sinusoid dataset, with 2 GPUs and a meta batch size of 16
```
python run_1d_maml.py --gpus 0,1 --nr_model 16 --learning_rate 0.0001 --dataset_name sinusoid
```
