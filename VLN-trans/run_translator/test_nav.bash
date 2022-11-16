name=My_test

flag="--vlnbert prevalent

      --submit 1
      --test_only 0

      --train validlistener
      --load /localscratch/zhan1624/VLN-speaker/snap/my_navigator2/state_dict/best
      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"
  

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=2 python r2r_src_navigator/train.py $flag --name $name