# for L in {512..16384..512}
# do
#     CUDA_VISIBLE_DEVICES=0 python testing_topk.py $L
# done

# for attn in performer #Q-chunking #performer #Q-chunking vanilla  
# do
#     for L in 2048 {2048..65536..2048}
#     do
#         CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking2  --K_chunk_size -1 --batch_size 1  --n_heads 12  --layer causal-self-attn  --head_size 64  --Q_chunk_size 1024 --topk 128  --backward   --n_queries $L  --n_keys $L  --attn_variant $attn
#     done
# done

# for attn in Q-chunking #Q-chunking # vanilla
# do
#     for L in 2048 {2048..65536..2048}
#     do
#         CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking  --K_chunk_size -1 --n_heads 0  --layer ff  --head_size 768  --backward   --n_queries 512  --Q_chunk_size 16384 --topk -1 --batch_size 512 --n_keys $L  --attn_variant $attn
#     done
# done

# for attn in vanilla #Q-chunking # topk-sparse #
# do
#     for L in 2048 {2048..65536..2048}
#     do
#         CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking  --n_heads 0  --layer ff  --head_size 768  --backward   --n_queries 512  --batch_size 512  --topk -1 --n_keys $L  --K_chunk_size -1 --Q_chunk_size 16384 --attn_variant $attn
#     done
# done


# for attn in performer # Q-chunking #vanilla #Q-chunking
# do
#     for L in 2048 {2048..32768..2048}
#     do
#         CUDA_VISIBLE_DEVICES=0 python benchmarking_multiple_layers.py  --output_dir out_benchmarking  --batch_size 1 --n_heads 12 --head_size 64  --ff_dim 3072 --n_layers 12  --causal --backward --Q_chunk_size_attn 1024 --Q_chunk_size_ff 4096 --topk_attn -1 --topk_ff -1  --n_queries $L --attn_variant $attn
#     done
# done

# for attn in vanilla #Q-chunking #vanilla #Q-chunking
# do
#     for L in 2048 {2048..4096..512}
#     do
#         CUDA_VISIBLE_DEVICES=0 python benchmarking_multiple_layers.py  --output_dir out_benchmarking3  --batch_size 1 --n_heads 12 --head_size 64  --ff_dim 3072 --n_layers 12  --causal --backward --Q_chunk_size_attn 1024 --Q_chunk_size_ff -1 --topk_attn 64 --topk_ff -1  --n_queries $L --attn_variant $attn
#     done
# done


# for C in 128 128 256  # 512 {512..8192..512}
# do
#     CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking_trade_off  --K_chunk_size -1 --batch_size 1  --n_heads 12  --layer no-mask-self-attn  --head_size 64  --Q_chunk_size $C --topk -1  --n_queries 65536  --n_keys 65536  --attn_variant Q-chunking
# done



# for attn in topk-sparse   # Q-chunking  
# do
#     for L in 2048 {2048..65536..2048}
#     do
#         CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking_sparse  --n_heads 0  --layer ff  --head_size 768  --backward   --n_queries 512  --batch_size 512  --topk 512 --n_keys $L  --K_chunk_size -1 --Q_chunk_size 16384 --attn_variant $attn
#     done
# done