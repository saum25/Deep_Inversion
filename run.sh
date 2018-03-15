#!/bin/bash
rm -rf models/fc8/training_log.txt # removes the training log file if present
echo
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Script Starts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% "
echo
echo
echo "================================================================================"
echo "Training Inversion Models for FC8 inversion [Feature Space and Input Space Loss]"
echo "================================================================================"

idx=14 # First model index
counter=1 # in bash there is no space between the variable and the assignment operator

for comp_layer_name in 'fc8' 'fc7' 'mp6' 'conv5' 'conv4' 'mp3' 'conv2' 'conv1'
do
 for input_loss_weight in 1e0 1e-1 1e-2 1e-3
 do
  echo
  echo "--------------------------------------------------------------"
  echo "Comparator layer: $comp_layer_name Weight: $input_loss_weight"
  echo "--------------------------------------------------------------"
  echo
  echo "******* Model Training Starts *********"

  THEANO_FLAGS=device=gpu0 python train.py jamendo_augment.npz ./models/fc8/jamendo_augment_gen_fc8_a${idx}.npz --no-augment --lr_init 0.001 --lr_decay 3 --comp_layer_name $comp_layer_name --w_inputloss $input_loss_weight
  #echo "jamendo_augment_gen_fc8_a${idx}.npz" 
  idx=$(($idx + $counter))
  echo "******* Model Training Ends ***********"
 done
done
echo 
echo "%%%%%%Script Ends%%%%%%"