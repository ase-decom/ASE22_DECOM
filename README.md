# ASE22_DECOM
This is a replication package for `Automatic Comment Generation via Multi-Pass Deliberation`. 
Our project is public at: <https://github.com/ase-decom/ASE22_DECOM>

## Content
1. [Get Started](#1-Get-Started)<br>
&ensp;&ensp;[1.1 Requirements](#11-Requirements)<br>
&ensp;&ensp;[1.2 Dataset](#12-Dataset)<br>
&ensp;&ensp;[1.3 Train and Test](#13-Train-and-Test)<br>
2. [Project Summary](#2-Project-Summary)<br>
3. [Model](#3-Model)<br>
4. [Experiments](#4-Experiments)<br>
&ensp;&ensp;[4.1 Dataset](#41-Dataset)<br>
&ensp;&ensp;[4.2 Research Question](#42-Research-Question)<br>
5. [Results](#5-Results)<br>
&ensp;&ensp;[5.1 RQ1: Comparison with Baseline](#51-RQ1-Comparison-with-Baseline)<br>
&ensp;&ensp;[5.2 RQ2: Component Analysis](#52-RQ2-Component-Analysis)<br>
&ensp;&ensp;[5.3 RQ3: Performance for Different Lengths](#53-RQ3-Performance-for-Different-Lengths)<br>
6. [Human Evaluation](#6-Human-Evaluation)<br>

## 1 Get Started
### 1.1 Requirements
* Hardwares: NVIDIA GeForce RTX 3060 GPU, intel core i5 CPU
* OS: Ubuntu 20.04
* Packages: 
  * python 3.8 (for running the main code)
  * pytorch 1.9.0
  * cuda 11.1
  * java 1.8.0 (for retrieving the similar code)
  * python 2.7 (for evaluation)

### 1.2 Dataset
DECOM is evaluated on [JCSD](https://github.com/xing-hu/TL-CodeSum) and [PCSD](https://github.com/EdinburghNLP/code-docstring-corpus) benchmark datasets. The structures of ```dataset/JCSD``` and ```dataset/PCSD``` are as follows:
* train/valid/test
  *  source.code: tokens of the source code
  *  source.comment: tokens of the source comments
  *  similar.code: tokens of the retrieved similar code
  *  similar.comment: tokens of the retrieved similar comments
  *  source.keywords: tokens of the code keywords(identifier names)


### 1.3 Train and Test
1. Go the ```src``` directory, process the dataset and generate the vocabulary:
```
cd src/
python build_vocab.py
```
2. Train DECOM model by performing a two-step training strategy:
```
python train_locally.py
python train_FineTune.py
```
3. Test DECOM model:
```
python prediction.py
```
4. Switch to python 2.7 environment and evaluate the performance of DECOM:
```
cd rencos_evaluation/
python evaluate.py
```


## 2 Project Summary
Deliberation is a common and natural behavior in human daily life. For example, when writing papers or articles, we usually first write drafts, and then iteratively polish them until satisfied.
In light of such a human cognitive process, we propose DECOM, which is a multi-pass deliberation framework for automatic comment generation. 
DECOM consists of multiple Deliberation Models and one Evaluation Model.
Given a code snippet, we first extract keywords from the code and retrieve a similar code fragment from a pre-defined corpus. Then, we treat the comment of the retrieved code as the initial draft and input it with the code and keywords into DECOM to start the iterative deliberation process. 
At each deliberation, the deliberation model polishes the draft and generates a new comment.

## 3 Model
<div align=center><img src="https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/framework.png" width="750"/></div><br>

This figure illustrates an overview of DECOM, which consists of three main stages: <br>
**(1)  Data initialization:** for extracting the keywords from the input code and retrieving the similar code-comment pair from the retrieval corpus;<br>
**(2)  Model training:** for leveraging a two-step training strategy to optimize DECOM;<br>
**(3)  Model prediction:** for generating the target comment of the new source code.<br>

<p align="middle">
  <img src="https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/deliberation.png" width="420" />
  <img src="https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/evaluation.png" width="300" /> 
</p>

DECOM contains *K* deliberation models and one evaluation model, where *K* is the maximum number of deliberations. Each deliberation model consisits of three different encoders (i.e. code encoder, keyword encoder, and comment encoder) and a decoder. The evaluation model contains a shared code encoder, a shared comment encoder, and an evaluator. At each deliberation, the deliberation model polishes the past draft and generates a new comment. The evaluation model calculates the quality score of the newly generated comment. This multi-pass process terminates when (1) the quality score of the new comments is no longer higher than the previous ones, or (2) the maximum number of deliberations *K* is reached.

## 4 Experiments
### 4.1 Dataset
We selected JCSD and PCSD benchmark datasets in our experiments. For the sake of fairness, we preprocess the two datasets strictly following Rencos. Specifically, we first split datasets into a training set, validation set, and test set in a consistent proportion of 8 : 1 : 1 for the Java dataset and 6 : 2 : 2 for the Python dataset. We use the *javalang* and *tokenize* libraries to tokenize the code snippet for JCSD and PCSD, respectively. We further split code tokens of the form CamelCase and snake_case to respective subtokens. For JCSD, we remove the exactly duplicated code-comment pairs in the test set. The specific statistics of the two preprocessed datasets are shown in the following table. <br>
<div align=center>
<img src="https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/dataset.png" width="420" />
</div>
<br>

### 4.2 Research Question
We address the following three research questions to evaluate the performance of DECOM:
- RQ1: How does the DECOM perform compared to the state-of-the-art comment generation baselines?
- RQ2: How does each individual component in DECOM contribute to the overall performance?
- RQ3: What’s the performance of DECOM on the data with different code or comment length?

For RQ1, we compare our approach with three categories of existing work on the comment generation task, using common metrics including BLEU, ROUGE-L, METEOR and CIDEr. IR-based baselines include LSI, VSM and NNGen. The NMT-based approaches include CODE-NN, TL-CodeSum and Hybrid-DRL. The Hybrid approaches include Rencos, Re2Com and EditSum.
<br><br>
For RQ2, to evaluate the contribution of core components, we obtain two variants: **(1) DECOM w/o Multi-pass Deliberation**, which removes the multi-pass deliberation and adopts the one-pass process to generate comments. **(2) DECOM w/o Evaluation Model**, which removes the evaluation model and takes the comment generated by the last deliberation model as the result. We train the two variants with the same experimental setup as DECOM and evaluate their performance on the test sets of JCSD and PCSD, respectively.
<br><br>
For RQ3, we analyze the performance of DECOM and best three baselines (i.e. Re2com, Rencos, and EditSum) on different lengths (i.e., number of tokens) of code and comments.
<br><br>


## 5 Results
### 5.1 RQ1: Comparison with Baseline
![](https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/RQ1.png)
<br>
The above table shows the comparison results between the performance of DECOM and other baselines, and the best performance is highlighted in bold. DECOM outperforms the state-of-the-art baselines in terms of all seven metrics on both two datasets. Compared to the best baseline Rencos, DECOM improves the performance of BLEU-4, ROUGE-L, METEOR, and CIDEr by 8.3%, 6.0%, 13.3%, and 10.5% on JCSD dataset, by 5.8%, 3.8%, 6.6%, and 6.3% on PCSD dataset, respectively.

### 5.2 RQ2: Component Analysis
![](https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/RQ2.png)
<br>
The above table presents the performances of DECOM and its two variants. Both the multi-pass deliberation and the evaluation model components have positive contributions to the performance of DECOM, where the multi-pass deliberation component contributes more to increasing the performance.

### 5.3 RQ3: Performance for Different Lengths
![](https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/RQ3.png)
<br>
This figure presents the performance of DECOM and the three baselines on JCSD and PCSD datasets with code and comments of different lengths, where the red lines denote the performance of DECOM. DECOM generally outperforms the best three baselines on different lengths of the input code snippets and the output comments, indicating its robustness. In particular for Java, DECOM can achieve much higher performance than others when the code snippets and comments are long.

## 6 Human Evaluation
We randomly select 100 code snippets from the test dataset (50 from JCSD and 50 from PCSD). By applying the best three baselines (i.e. Re2com, Rencos, and EditSum) and DECOM, we obtain 400 generated comments as our evaluating subjects. Each participant is asked to rate each commentfrom the three aspects: 
- Naturalness: reflects the fluency of generated text from the perspective of grammar; 
- Informativeness: reflects the information richness of generated summaries; 
- Usefulness: reflects how can generated summaries help developers.
<br>All three scores are integers, ranging from 1 to 5. Higher score means more positive.<br>
![](https://github.com/ase-decom/ASE22_DECOM/blob/master/diagrams/HE.png)
<br>
This figure exhibits the results of human evaluation by showing the violin plots depicting the naturalness, informativeness, and usefulness of different models. Each violin plot contains two parts, i.e., the left and right parts reflect the evaluation results of models on the JCSD dataset and PCSD dataset. The box plots in the violin plots present the distribution of data and the red triangles mean the average scores of the three aspects.<br>

Overall, DECOM is better than all baselines in three aspects. The average score for naturalness, informativeness, and usefulness of our approach are 4.24, 3.43, and 3.25, respectively, on the JCSD dataset. On the PCSD dataset, our approach gets the average score of 4.05, 2.96, and 2.87 in terms of naturalness, informativeness, and usefulness.
