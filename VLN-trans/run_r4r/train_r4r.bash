name=r4r_navigator

flag="--vlnbert prevalent
      --test_only 0
      --train auglistener
      --aug /egr/research-hlr/joslin/r2r/data/prevalent_aug_new_fine.json
      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 1e-5
      --iters 600000
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
      --accumulateGrad
     "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=4 python r4r_src/train.py $flag --name $name