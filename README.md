# Explanation-based Finetuning

Official implementation for our paper ["Explanation-based Finetuning Makes Models More Robust to Spurious Correlation"](https://arxiv.org/abs/2305.04990), ACL 2023.

## Getting started
Create a new conda environment using environment.yml. The env is named "ex-ft" by default.
```
conda env create -f environment.yml
```

## Download data
The following script downloads 4 datatset to a specified directory path. This includes CREAK, e-SNLI, COMVE, SBIC, all of which have free-form text explanations.
```
sh script/download_data.sh DIR_PATH
```

## Usage
### 1. Prepare data for finetuning
From the original datasets, we construct induce biased data with a spurious correlation for finetuning. The following script preprocesses the a biased e-SNLI dataset by skewing long sentences towards the Positive class. We apply a perfect 1.0 class bias and store output in `res`:
```
python3 construct_data.py --task_name esnlo --bias length --bias_strength=1.0 --data_dir DIR_PATH/data --output_dir DIR_PATH/res
```
Parameters available:
* `--task_name`: {creak,comve,esnli,sbic}
* `--bias`: {unbiased,length,present,cluster,plural}. We also include other task-specific biases in the script.
* `--bias_strength`: a float number between 0 and 1
* `--permute`: (Optional) To randomly permute the explanations, which is an ablation in our paper
* `--expl_temp`: (Optional) To apply a descriptive template for the explanation in the input

### 2. Finetune
a. For calling the OpenAI API to finetune a model:
```
python3 finetune_openai.py --api_key <YOUR_API_KEY> --filename esnli_present_finetuned_trainAdvanced_filterBias_100bias_1000train.csv --model davinci
```
Afterwards, please store the ID of your finetuned model to `openai_model_dict.py` following the template.

b. For finetuning other model families such as BART, T5, or OPT:
```
# T5 or BART
python3 finetune_t5_bart.py --task_name=esnli --data_dir=DIR_PATH/data --output_dir=DIR_PATH/res --cache_dir=CACHE_PATH --bias=length --model_type=t5-base --num_epochs=5 --train_size=100 --with_expl --verbose

# OPT
python3 finetune_opt.py --task_name=esnli --data_dir=DIR_PATH/data --output_dir=DIR_PATH/res --cache_dir=CACHE_PATH --bias=length --model_type=facebook/opt-350m --num_epochs=5 --train_size=100 --with_expl --verbose
```
Parameters available:
* `--model_type`: "t5-{small,base,large,xl,xxl}", "facebook/bart-{base,large}", or "facebook/opt-{125m,350m,1.3b,6.7b}"
* `--num_epochs`: how many epochs to train
* `--train_size`: how many data points for training
* `--with_expl`: (Optional) finetune with explanations
* `--verbose`: (Optional) print detailed logs to stdout
* `--cache_dir`: (Optional) cache directory for Huggingface models

For non-OpenAI models, predictions on the Test set are saved in `DIR_PATH/res/`.

### 3. Evaluation on finetuned OpenAI models

Do inference over the Test set and report Accuracy, F1, and correlation with the biased feature:
```
python3 evaluate.py --task_name esnli --bias length --method finetuned --with_expl  --data_dir DIR_PATH/data --output_dir DIR_PATH/res
```

Last 3 lines of expected output:
```
Using specified model: davinci:ft-ccb-lab-members-2022-12-26-20-28-30
inferencing
result saved to  /content/test_esnli_present_adv_finetuned.csv
acc:0.864, f1:0.8650793650793651
corr:  0.20473490198559788
```

## Citation
If you find our work helpful, please cite:
```bibtex
@inproceedings{ludan-et-al-2023-explanation-based-finetuning,
  author = {Ludan, Josh Magnus and Meng, Yixuan and Nguyen, Tai and Shah, Saurabh and Lyu, Qing and Apidianaki, Marianna and Callison-Burch, Chris},
  title = {Explanation-based Finetuning Makes Models More Robust to Spurious Cues},
  booktitle={Proceedings of the The 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)},
  address={Toronto, Canada}
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
