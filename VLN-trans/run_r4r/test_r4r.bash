name=r4r_test

flag="--vlnbert prevalent
      --submit 1
      --test_only 0
      --train validlistener
      --load /localscratch/zhan1624/VLN-speaker/snap/r4r_navigator/state_dict/test
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
