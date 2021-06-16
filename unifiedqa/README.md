### T5/unifiedQA experiments using top-k attention at FF layers

- download the pre-processed data from [unifiedqa](https://github.com/allenai/unifiedqa) as follows:
```bash
bash download.sh
```  
<br/>

- convert to `.jsonl`
```bash
for k in commonsenseqa squad1_1 squad2 boolq narrativeqa mctest_corrected_the_separator race_string openbookqa arc_easy arc_hard ai2_science_middle ai2_science_elementary ropes
do
    python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data/$k --dataset $k --split train --cores 8 --add_special_words
    python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data/$k --dataset $k --split dev --cores 8 --add_special_words
    python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data/$k --dataset $k --split dev --cores 8
done
```
`cores` are number of cpu cores you want to use for multiprocesssing. You can add more datasets from `data/preprocessed_datasets`.

- making 0-shot predictions from unifiedQA by directly using top-k attention at FF layers
```bash
CUDA_VISIBLE_DEVICES=0,1 python run_unifiedqa.py --model_name_or_path allenai/unifiedqa-t5-11b --file_path data/<DATASET_NAME>/combined_dev_unifiedqa.jsonl --output_dir out_preds_unifiedqa_base/<DATASET_NAME>_-1 --eval_batch_size 16 --ff_proj_topk -1
```
where `<DATASET_NAME>` is name of the dataset such as `openbookqa`, etc. Instead of `--ff_proj_topk -1` (off) you can use something like 256, 512, etc. Note that layers will be partitioned across gpus by default which can be helpful for large models.


- _interpretation of em/f1_ : these are computed using squad1.1 eval script and might differ from the official metrics! Similar to squad1.1, max metric is used in case multiple answers are provided in the corresponding `<split>_ans.jsonl` file inside `data/preprocessed_datasets/<DATASET_NAME>`.


### finetuning T5 using top-k attention at FF layers

- install transformers from source
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
<br/>

- training  
```bash
CUDA_VISIBLE_DEVICES=0 python run_summarization.py     --model_name_or_path t5-base     --do_train     --learning_rate 5e-5     --lr_scheduler_type constant     --max_grad_norm 1     --evaluation_strategy no   --overwrite_output_dir   --group_by_length  --save_total_limit 1   --save_steps 500       --max_steps 3001     --train_file ./data/<DATASET_NAME>/combined_train_t5.jsonl    --text_column input     --summary_column ans     --output_dir out_<DATASET_NAME>_-1   --logging_dir out_<DATASET_NAME>_-1/log  --per_device_train_batch_size=128  --gradient_accumulation_steps 4   --logging_steps 100   --local_files_only    --ff_proj_topk -1  --Q_chunk_size -1
```
where `<DATASET_NAME>` is name of the dataset such as `openbookqa`, etc. To resume training from the last checkpoint inside `--output_dir`, simply run the same command but without `--overwrite_output_dir`. Again, instead of `--ff_proj_topk -1` (off) you can use something like 256, etc. Please refer to the appendix of our paper for hyperparameters.

- inference  
```bash
CUDA_VISIBLE_DEVICES=0 python run_unifiedqa.py --model_name_or_path out_<DATASET_NAME>_-1  --file_path data/<DATASET_NAME>/combined_dev_t5.jsonl --output_dir out_preds_t5_base/<DATASET_NAME>_-1 --eval_batch_size 32 --ff_proj_topk -1
```
   
<br/>

NOTES:
* while pre-processing datasets, `--add_special_words` flag produces data for T5 whereas without it produces for unifiedqa.
* if `--local_files_only` flag stops you from downloading the pre-trained model, try again without it.
* for each training its tensorboard `log` dir is inside the `--output_dir` dir.
* training uses ADAM as we couldn't get good results with Adafactor.
