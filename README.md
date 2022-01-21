# GP Evaluation Profiling
This repository provides a means to profile (i.e., benchmark) 
the *evaluation* methodologies given by some genetic programming 
(GP) tools. Any evolutionary mechanisms provided by a GP tool 
are *not* included when profiling——only mechanisms for calculating
"fitness."

This repository was originally created for the <insert_paper_title>
paper, which aimed to compare the runtime performance of an
initial FPGA-based hardware architecture for evaluating GP 
programs with that of certain GP software tools.

## Included Tools

At this point in time, a means for profiling for the following GP tools 
is given:

- [DEAP](https://github.com/DEAP/deap) - for the original paper, 
click [here](http://vision.gel.ulaval.ca/~cgagne/pubs/deap-gecco-2012.pdf).
- [TensorGP](https://github.com/AwardOfSky/TensorGP) - for the original paper,
click [here](https://cdv.dei.uc.pt/wp-content/uploads/2021/04/baeta2021tensorgp.pdf).
- [Operon](https://github.com/heal-research/operon) - for the original paper,
click [here](https://dl.acm.org/doi/pdf/10.1145/3377929.3398099).


## Installation instructions

The following has been verified via the Ubuntu and CentOS operating systems.
It is likely that other Linux distributions are 
supported, and it is plausible that Windows operating systems
are supported, but it is unlikely that MacOS is readily supported.

### Prerequisites
- Ensure that some Conda package management system 
(e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) 
is installed on the relevant machine.
- Clone this repository onto the relevant machine.

Upon cloning the repository, set up the relevant Conda environment
and install the relevant tools by executing the following within
a shell program, after having navigated to the repository directory
within the shell:

```
conda env create -f environment.yml
conda activate gp-eval-profile
bash install.sh
```

## Profiling
By default, the repository already contains some profiling results,
scattered throughout the `experiment/results` directory. 
(TODO: talk more about this.)

After successfully completing installation, you may run the entire profiling
suite by executing the following within a shell program, after having navigated 
to the repository directory within the shell:

```
cd experiment
bash run.sh
```

After the `run.sh` script fully executes, to view some relevant statistics,
fully run the Jupyter Notebook given by the `tools/stats.ipynb` file.

**TODO: Finish**
