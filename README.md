# The EarlyBird Gets the WORM: Heuristically Accelerating Early Bird Convergence
This is the codebase for the paper "The EarlyBird Gets the WORM: Heuristically Accelerating EarlyBird Convergence", accepted to the Efficient Natural Language and Speech Processing Workshop at NeurIPS 2024.

The workshop can be found [here](https://neurips2024-enlsp.github.io/), and a link to the paper can be found [here](https://arxiv.org/abs/2406.11872).

## Prerequisite Libraries
Use the requirements.txt file to install the necessary libraries.

## Reproducing Results

### CNN Experiments
Run the cnn.py file to reproduce results for the ResNet-18 experiments.\\
Run the vgg.py file to reproduce results for the VGG-11 experiments.

- To run naive EarlyBird, type in "standard" as the command line argument
- To run Gradient Clipped EarlyBird, type in "gradclip" as the command line argument
- To run Greedy Clipped EarlyBird, type in "greedy" as the command line argument
- To run WORM, type in "worm" as the command line argument

### Transformer Experiments
Run the bert.py file to reproduce results for the BERT experiments.\\
Run the gemma-2b.py file to reproduce results for the Gemma-2B experiments.

- To run naive EarlyBird, type in "standard" as the command line argument
- To run WORM, type in "worm" as the command line argument


## Citations
https://github.com/GATECH-EIC/Early-Bird-Tickets

https://github.com/google/gemma_pytorch
