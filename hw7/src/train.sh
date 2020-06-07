
for i in {1..10}
do
    for l2p in 0.00001 0.000001 0.0000001 
    do
        for lr in 0.001
        do 
            for nt in 32 64 128
            do
                python3 main.py \
                    -iter 10000 \
                    -seed $i \
                    -validation_split 0.1 \
                    -num_topic $nt \
                    -use_bpr 1 \
                    -lr $lr \
                    -l2_penalty $l2p \
                    -use_wandb 1

                rm -rf wandb/                
            done
        done
    done

done