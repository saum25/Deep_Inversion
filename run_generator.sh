#!/bin/bash
rm -rf models/fc8/performance_log.txt # removes the training log file if present
echo
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Script Starts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% "
echo
echo
echo "================================================================================"
echo "Performance checking of the Inversion Models for FC8 inversion [Feature Space and Input Space Loss]"
echo "================================================================================"

# extra case for arch idx=10.
THEANO_FLAGS=device=gpu1 python generate.py jamendo_augment.npz ./models/fc8/jamendo_augment_gen_fc8_a10.npz --featloss

for idx in {14..45}
do
  echo
  echo "--------------------------------------------------------------"
  echo "Model Idx: $idx"
  echo "--------------------------------------------------------------"
  echo
  echo "******* Model Performance Evaluation Starts *********"

  THEANO_FLAGS=device=gpu1 python generate.py jamendo_augment.npz ./models/fc8/jamendo_augment_gen_fc8_a${idx}.npz --featloss
  #echo "jamendo_augment_gen_fc8_a${idx}.npz" 
  echo "******* Model Training Ends ***********"
done
echo 
echo "%%%%%%Script Ends%%%%%%"