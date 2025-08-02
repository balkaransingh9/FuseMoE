#!/bin/bash

# Define the range of hyperparameters
test_rs_values=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
test_noise_values=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.2 0.25 0.3)

# Loop over each combination of hyperparameters
for test_rs in "${test_rs_values[@]}"
do
    for test_noise in "${test_noise_values[@]}"
    do
        echo "Running sim_plots.py with test_rs=${test_rs} and test_noise=${test_noise}"
        python sim_plots.py --test_rs ${test_rs} --test_noise ${test_noise}
    done
done
