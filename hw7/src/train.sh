
for i in {1..1}
do
    for lr in 0.01 0.001 0.0001
    do
        for l2p in 0.0001 0.00001 0.000001 0.0000001
        do 
            for nt in 8 16 32 64 128 256
            do

                python main.py \
                    -iter 10000 \
                    -seed $i \
                    -validation_split 0.1 \
                    -num_topic $nt \
                    -use_bpr 1 \
                    -lr $lr \
                    -l2_penalty 0.000005 $l2p \
                    -use_wandb 1
                
            done
        done
    done

done