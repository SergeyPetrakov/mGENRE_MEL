# mGENRE_MEL
This repository contains files and materials related to multilingual entity linking task (MEL), especially basing on the mGENRE model since it is SOTA model. We consider MEL as a part of big knowledge base question answering (KBQA) that is called information retrieval part. Within this part we retrieve entities. Basing on them we can make queries to knowledge base. Thus, we obtain KBQA system

This repository is based on https://arxiv.org/abs/2103.12528 where mGENRE model is proposed.


## Quickstart

In command line run the folllowing six commands:

1) set new environment `conda create --name new_environment`
2) activate new environment `conda activate new_environment`
3) clone repository `git clone https://github.com/SergeyPetrakov/mGENRE_MEL`
4) go to the cloned repository `cd mGENRE_MEL`
5) install requirements `pip install -r requirements.txt`
6) launch jupyter notebook (for example: `jupyter notebook --ip 0.0.0.0 --port=7643 --no-browser --allow-root&`)
and open mgenre_final there (paste in browser web link and open file)

Experiments are provided in `mGENRE_and_Uncertainty_Estimation.ipynb` file:
 - Quickstart
 - Experiments with uncertainty estimation using metrics: `Entropy` on a single model, `Delta`, `Maxprob`, `Predicted entropy`, `Expected entropy` and `BALD` on `Simple Questions`, `RuBQ 2.0` and `Meusli-9` datasets

Necessary data and pretrained model jupyter notebook contain in cell `data`. If you once installed it you do not really need run this cell further.
We strongly recommend to follow the original article and repository to understand how everything works from the inside.





