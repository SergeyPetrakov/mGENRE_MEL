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

Necessary data and pretrained model jupyter notebook contain in cell `data`. If you once installed it you do not really need run this cell further.
We strongly recommend to follow the original article and repository to understand how everything works from the inside.

## Experiments

Experiments are provided in `mGENRE_and_Uncertainty_Estimation.ipynb` file, you can find there:
 - Quickstart
 - Experiments with uncertainty estimation using metrics: `Entropy` on a single model, `Delta`, `Maxprob`, `Predicted entropy`, `Expected entropy` and `BALD` on `Simple Questions`, `RuBQ 2.0` and `Meusli-9` datasets
 
 ## Results
 
 ### Illustrations and tables

File `3_dataset_experiments.pdf` contains rejection curves - visual illustration of uncertainty estimation integration into mGENRE quality assessment. As a numerical measure of unsertainty estimation quality were added two types of ares under rejection curve: absolute (equals to the whole area under curve) and comparative (equals to the area that is higher than quality received on all samples of dataset).

### Summary of results

- Experiments demonstrate that uncertainty quantification could be efficiently used for the task of entity linking in case of mGENRE model. This is especially obvious for English, German, Russian. Although there are cases when uncertainty estimation does not help, for example for Serbian and in some cases for Javanese, Japanese and Persian.
- The structure of dataset also matters, because the structure differs, for example model may perform better on RUBQ 2.0 and Simple Questions because they consist of short questions of widespread languages (English and Russian).
- There is no significant leader in terms of metric of uncertainty, but every time both for absolute area under rejection curve and area under curve added by uncertainty integration predicted entropy showed the best results among all metrics almost each time as for information received on aggregated AUC data.
- Thus, mGENRE performs well on the task of entity linking even in multilingual case. But end-to-end question answering system could not be realized using only mGENRE, even if we help it with such strong NER as Stanza.

### Conclusion
I can say that reached initial objectives. I conducted many experiments on entity linking and question answering task for mGENRE model, observed different datasets, types of questions (with single answer or potentially not one answer), checked different languages, quantified uncertainty using various metrics, such as entropy, maxprob, delta, predicted entropy, expected entropy and BALD. Provided quantifiable and comparable results using area under rejective curve approach.
Talking about the ways how one can take benefits from it, I can say that method of voting algorithms can be used, when we take some algorithms, mark answers they give with some uncertainty measure and choose the answer of the most confident algorithm among all.



