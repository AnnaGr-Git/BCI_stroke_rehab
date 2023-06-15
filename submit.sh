
#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J TestTraining
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s213637@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/f7/e/165914/BCI/BCI_stroke_rehab/gpu_%J.out
#BSUB -e /zhome/f7/e/165914/BCI/BCI_stroke_rehab/gpu_%J.err
# -- end of LSF options --

module unload cuda
nvidia-smi

# Load the cuda module
module load cuda/11.2

source BCI_env/bin/activate

python3 src/models/finetune_1D-CNN_physionet.py