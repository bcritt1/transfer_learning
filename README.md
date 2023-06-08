# Huggingface Transfer Learning Workflow

This repo contains two simple files to fine-tune a transformers model on hugging face with transfer learning from a domain-specific corpus of 
.txt files.

## File Overview

The files consist of:

1. [training.py](training.py): Reads a collection of txt files from a directory, creates a model of them, and then uses this model to update an LLM like BERT.
2. [training.sbatch](training.sbatch): Creates a batch job for training.py.

## Usage instructions

### Setting up and Connecting to Sherlock

1. Before we log onto Sherlock, let's make sure we're going to have everything we need there and move inputs/corpus onto Sherlock. For info on transferring data to Sherlock, see:
[https://www.sherlock.stanford.edu/docs/storage/data-transfer/](https://www.sherlock.stanford.edu/docs/storage/data-transfer/). [rsync](https://www.sherlock.stanford.edu/docs/storage/data-transfer/#rsync) is probably the best program for
this, but if you prefer another, go with that. For rsync, you'd use the command 
```bash
rsync -a ~/path/to/local/data yourSUNetid@login.sherlock.stanford.edu:/scratch/users/$USER/corpus/
```
You'll need to tweak the local path because I don't know where your files are located, but the remote path (after the ":") should work fine to get your corpus into scratch, a fast storage system where it's best to do file 
reading/writing.

2. Now we can log onto Sherlock using ssh in the Terminal program on Mac[^1]. with the syntax: 
```bash
ssh yourSUNetID@sherlock.stanford.edu
```
### File Management

3. Once we're logged on, we want to put these files on Sherlock:
```bash
git clone https://github.com/bcritt1/transfer_learning.git
```
This will create a directory in your home space on Sherlock called "transfer_learning" with all the files in this repository.

4. Let's also make three directories for the outputs of our process:
```bash
mkdir out err /scratch/users/$USER/outputs
```
### Running Code

5. Now, let's move into our new directory
```bash
cd transfer_learning
```
and submit our sbatch file to slurm, Sherlock's job scheduler: 
```
sbatch training.sbatch
```
You can watch your program run with
```
watch squeue -u $USER
```
When it finishes running, you should see your outputs as .csv and .json files in the outputs/ 
directory on scratch. This data can then be used as an input for other processes, or analyzed on its own.

## Code Explanation

The above walkthrough is designed to be as easy as possible to execute. If it works for you and you don't want to know about the code, you may not need to read this. If you want to know more, or need to tweak something, this section will 
help.

### training.sbatch

A batch script that gives [slurm](https://slurm.schedmd.com/pdfs/summary.pdf), Sherlock's job scheduler, instructions on how to set up for and run our python code. Everything that starts with a "#" are directives to slurm, and everything 
after are commands we're executing in the terminal on Sherlock.
```bash
#!/usr/bin/bash
#SBATCH --job-name=training					# gives the job an arbitrary name, which will be used in queue and for err and out files
#SBATCH --output=/home/users/%u/out/training.%j.out		# directs output files to an out/ directory in the user's home
#SBATCH --error=/home/users/%u/err/training.%j.err		# directs error files to an err/ directory in the user's home
#SBATCH --partition=gpu						# sends the job to the gpu partition, which is best suited for ML tasks
#SBATCH --nodes=1						# job uses 1 node
#SBATCH --ntasks-per-node=1					# number of cores on a single node 	
#SBATCH --cpus-per-task=4					# cpus given to each task
#SBATCH --time=2:00:00						# two hour time limit
#SBATCH --gres=gpu:1						# one GPU requested
#SBATCH --mem=64GB						# 64 gb RAM requested
	
module load python/3.9.0										# load sherlock's latest version of python
pip3 install transformers										# load transformers for huggingface functionality
pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html		# install nightly version of torch for ML
python3 training.py											# run the python script
```

### training.py

The python file contains pretty frequent in-line documentation, which you can check out using either ```cat training.py``` or ```nano training.py``` for more detail. As an outline, the script loads a BERT language model and tokenizer for huggingface, 
uses them on 
a 
corpus of files that I route it to automatically using environment variables (it reads your username from Sherlock and uses that to find your files) to perform word tokenization, model the reference corpus, and combine it with the BERT model.  
If you have issues, contact [us](mailto:srcc-support@stanford.edu) and ask for Brad Rittenhouse.

 #### Notes

[^1]: The syntax would be the same if you use Terminal on Linux or Windows Subsystem for Linux [(WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). Using other programs is possible, but documenting them here would be 
impossible. 
