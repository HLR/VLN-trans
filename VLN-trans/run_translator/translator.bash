name=translator

flag="--vlnbert prevalent
      --test_only 0
      --aug /localscratch/zhan1624/VLN-speaker/r2r_src_helper5/speaker_data/aug.json
      --train speaker
      --features places365
      --maxAction 15
      --batchSize 16
      --feedback samplex
      --lr 1e-5
      --iters 600000
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
     "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=4 python r2r_src_helper5/train.py $flag --name $name