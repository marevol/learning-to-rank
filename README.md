# Learning To Rank (LTR)

A curated list of research papers, software, datasets, and resources for Learning to Rank (LTR), also known as machine-learned ranking.

## What is Learning to Rank?

Learning to Rank (LTR) is a machine learning approach for constructing ranking models from training data. LTR has been particularly successful in information retrieval, web search, and recommendation systems where the goal is to rank a set of items according to their relevance to a given query.

### LTR Approaches

LTR algorithms can be categorized into three main approaches:

- **Pointwise Approach**: Treats ranking as a regression or classification problem, predicting a relevance score for each document independently.
  - Examples: McRank, Prank, OC SVM

- **Pairwise Approach**: Focuses on learning the relative order between pairs of documents. The goal is to minimize the number of incorrectly ordered pairs.
  - Examples: RankNet, RankBoost, RankSVM, LambdaRank, LambdaMART

- **Listwise Approach**: Directly optimizes ranking metrics (e.g., NDCG, MAP) by considering the entire list of documents.
  - Examples: ListNet, ListMLE, AdaRank, SoftRank, LambdaMART

## Table of Contents

- [Papers](#papers)
  - [2003-2020](#2003)
  - [2021-2025](#2021)
- [Software](#software)
- [Dataset](#dataset)
- [Others](#others)

## Papers

### 2003

- Freund, Yoav, et al. "[An efficient boosting algorithm for combining preferences.](https://www.jmlr.org/papers/volume4/freund03a/freund03a.pdf)" Journal of machine learning research 4.Nov (2003): 933-969.

### 2005

- Burges, Chris, et al. "[Learning to rank using gradient descent.](https://www.researchgate.net/profile/Christopher_Burges/publication/221345726_Learning_to_Rank_using_Gradient_Descent/links/00b49518c11a6cbcb8000000.pdf)" Proceedings of the 22nd international conference on Machine learning. 2005.

### 2007

- Xu, Jun, and Hang Li. "[Adarank: a boosting algorithm for information retrieval.](http://www.bigdatalab.ac.cn/~junxu/publications/SIGIR2007_AdaRank.pdf)" Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 2007.
- Yue, Yisong, et al. "[A support vector method for optimizing average precision.](http://www.cs.cornell.edu/~tj/publications/yue_etal_07a.pdf)" Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 2007.
- Geng, Xiubo, et al. "[Feature selection for ranking.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.480.3451&rep=rep1&type=pdf)" Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 2007.
- Tsai, Ming-Feng, et al. "[FRank: a ranking method with fidelity loss.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2006-155.pdf)" Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 2007.
- Cao, Zhe, et al. "[Learning to rank: from pairwise approach to listwise approach.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.333.4334&rep=rep1&type=pdf)" Proceedings of the 24th international conference on Machine learning. 2007.
- Burges, Christopher J., Robert Ragno, and Quoc V. Le. "[Learning to rank with nonsmooth cost functions.](http://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf)" Advances in neural information processing systems. 2007.
- Zheng, Zhaohui, et al. "[A regression framework for learning ranking functions using relative relevance judgments.](https://www.cc.gatech.edu/~zha/papers/fp086-zheng.pdf)" Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 2007.
- Qin, Tao, et al. "[Ranking with multiple hyperplanes.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.3093&rep=rep1&type=pdf)" Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 2007.

### 2008

- Amini, Massih Reza, Tuong Vinh Truong, and Cyril Goutte. "[A boosting algorithm for learning bipartite ranking functions with partially labeled data.](http://ama.liglab.fr/~amini/Publis/SemiSupRanking_sigir08.pdf)" Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.
- Xu, Jun, et al. "[Directly optimizing evaluation measures in learning to rank.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.156.8455&rep=rep1&type=pdf)" Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.
- Veloso, Adriano A., et al. "[Learning to rank at query-time using association rules.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.144.5407&rep=rep1&type=pdf)" Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.
- Duh, Kevin, and Katrin Kirchhoff. "[Learning to rank with partially-labeled data.](http://www.cs.jhu.edu/~kevinduh/papers/duh08sigir.pdf)" Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.
- Guiver, John, and Edward Snelson. "[Learning to rank with softrank and gaussian processes.](https://dl.acm.org/doi/abs/10.1145/1390334.1390380)" Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.
- Zhou, Ke, et al. "[Learning to rank with ties.](https://dl.acm.org/doi/abs/10.1145/1390334.1390382)" Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.
- Geng, Xiubo, et al. "[Query dependent ranking using k-nearest neighbor.](https://andrewoarnold.com/fp025-geng.pdf)" Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.

### 2009

- Lease, Matthew. "[An improved markov random field model for supporting verbose queries.](https://www.ischool.utexas.edu/~ml/papers/lease-sigir09.pdf)" Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.
- Aslam, Javed A., et al. "[Document selection methodologies for efficient and effective learning-to-rank.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.2220&rep=rep1&type=pdf)" Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.
- Donmez, Pinar, Krysta M. Svore, and Christopher JC Burges. "[On the local optimality of LambdaRank.](http://www.cs.cmu.edu/afs/.cs.cmu.edu/Web/People/pinard/Papers/sigirfp092-donmez.pdf)" Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.
- Cummins, Ronan, and Colm O'Riordan. "[Learning in a pairwise term-term proximity framework for information retrieval.](https://www.researchgate.net/profile/Ronan_Cummins/publication/221299338_Learning_in_a_Pairwise_Term-Term_Proximity_Framework_for_Information_Retrieval/links/0912f50fa61d97a283000000.pdf)" Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.
- Banerjee, Somnath, Soumen Chakrabarti, and Ganesh Ramakrishnan. "[Learning to rank for quantity consensus queries.](https://www.cse.iitb.ac.in/~soumen/doc/sigir2009q/QCQ.pdf)" Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.
- Sun, Zhengya, et al. "[Robust sparse rank learning for non-smooth ranking measures.](https://www.researchgate.net/profile/Zhengya_Sun/publication/221301280_Robust_Sparse_Rank_Learning_for_Non-Smooth_Ranking_Measures/links/551bd2b20cf2909047b96a96.pdf)" Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.

### 2010

- Burges, Christopher JC. "[From ranknet to lambdarank to lambdamart: An overview.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)" Learning 11.23-581 (2010): 81.
- Svore, Krysta M., Pallika H. Kanani, and Nazan Khan. "[How good is a span of terms? Exploiting proximity to improve web retrieval.](https://pallika.github.io/files/fp728-svore.pdf)" Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval. 2010.
- Wang, Lidan, Jimmy Lin, and Donald Metzler. "[Learning to efficiently rank.](http://lintool.github.io/NSF-projects/IIS-1144034/publications/Wang_etal_SIGIR2010.pdf)" Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval. 2010.
- Gao, Wei, et al. "[Learning to rank only using training data from related domain.](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5600&context=sis_research)" Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval. 2010.
- Bagherjeiran, Abraham, Andrew O. Hatch, and Adwait Ratnaparkhi. "[Ranking for the conversion funnel.](https://dl.acm.org/doi/abs/10.1145/1835449.1835476)" Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval. 2010.

### 2011

- Wang, Lidan, Jimmy Lin, and Donald Metzler. "[A cascade ranking model for efficient ranked retrieval.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.357.4790&rep=rep1&type=pdf)" Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 2011.
- Dai, Na, Milad Shokouhi, and Brian D. Davison. "[Learning to rank for freshness and relevance.](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/Dai2011.pdf)" Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 2011.
- Ganjisaffar, Yasser, Rich Caruana, and Cristina Videira Lopes. "[Bagging gradient-boosted trees for high precision, low variance ranking models.](https://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/materials/bagging_lmbamart_jforests.pdf)" Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 2011.
- Cai, Peng, et al. "[Relevant knowledge helps in choosing right teacher: active query selection for ranking adaptation.](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5597&context=sis_research)" Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 2011.
- Chapelle, Olivier, and Yi Chang. "[Yahoo! learning to rank challenge overview.](http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf?WT.mc_id=Blog_MachLearn_General_DI)" Proceedings of the learning to rank challenge. 2011.

### 2012

- Wang, Lidan, Paul N. Bennett, and Kevyn Collins-Thompson. "[Robust ranking models via risk-sensitive optimization.](http://www.cs.cmu.edu/afs/cs/Web/People/pbennett/papers/wang-et-al-sigir-2012.pdf)" Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval. 2012.
- Severyn, Aliaksei, and Alessandro Moschitti. "[Structural relationships for large-scale learning of answer re-ranking.](http://dit.unitn.it/moschitti/articles/2012/SIGIR2012.pdf)" Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval. 2012.
- Niu, Shuzi, et al. "[Top-k learning to rank: labeling, ranking and evaluation.](http://www.chinakdd.com/include/ueditor/jsp/upload/20120910/71891347254381370.pdf)" Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval. 2012.

### 2013

- Paik, Jiaul H. "[A novel TF-IDF weighting scheme for effective ranking.](http://www.tyr.unlu.edu.ar/tallerIR/2014/papers/novel-tfidf.pdf)" Proceedings of the 36th international ACM SIGIR conference on Research and development in information retrieval. 2013.
- Wang, Hongning, et al. "[Personalized ranking model adaptation for web search.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.309.2388&rep=rep1&type=pdf)" Proceedings of the 36th international ACM SIGIR conference on Research and development in information retrieval. 2013.
- Raiber, Fiana, and Oren Kurland. "[Ranking document clusters using markov random fields.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.474.781&rep=rep1&type=pdf)" Proceedings of the 36th international ACM SIGIR conference on Research and development in information retrieval. 2013.

### 2015

- Grbovic, Mihajlo, et al. "[Context-and content-aware embeddings for query rewriting in sponsored search.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.707.9622&rep=rep1&type=pdf)" Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. 2015.
- Severyn, Aliaksei, and Alessandro Moschitti. "[Learning to rank short text pairs with convolutional deep neural networks.](http://eecs.csuohio.edu/~sschung/CIS660/RankShortTextCNNACM2015.pdf)" Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. 2015.
- Vulić, Ivan, and Marie-Francine Moens. "[Monolingual and cross-lingual information retrieval models based on (bilingual) word embeddings.](https://www2.kbs.uni-hannover.de/fileadmin/institut/pdf/webscience/2016-17/papers/got3.pdf)" Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. 2015.

### 2016

- Ustinovskiy, Yury, et al. "[An optimization framework for remapping and reweighting noisy relevance labels.](https://www.researchgate.net/profile/Pavel_Serdyukov/publication/305081472_An_Optimization_Framework_for_Remapping_and_Reweighting_Noisy_Relevance_Labels/links/5a9d3dc045851586a2aec23f/An-Optimization-Framework-for-Remapping-and-Reweighting-Noisy-Relevance-Labels.pdf)" Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.
- de Sá, Clebson CA, et al. "[Generalized BROOF-L2R: A general framework for learning to rank based on boosting and random forests.](https://dl.acm.org/doi/10.1145/2911451.2911540)" Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.
- Wang, Xuanhui, et al. "[Learning to rank with selection bias in personal search.](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45286.pdf)" Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.

### 2017

- Ibrahim, Muhammad, and Mark Carman. "[Comparing pointwise and listwise objective functions for random-forest-based learning-to-rank.](https://users.monash.edu.au/~mcarman/papers/ibrahim_TOIS2016_draft.pdf)" ACM Transactions on Information Systems (TOIS) 34.4 (2016): 1-38.
- Chen, Ruey-Cheng, et al. "[Efficient cost-aware cascade ranking in multi-stage retrieval.](http://culpepper.io/publications/cgbc17-sigir.pdf)" Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2017.
- Xiong, Chenyan, et al. "[End-to-end neural ad-hoc ranking with kernel pooling.](https://arxiv.org/pdf/1706.06613.pdf)" Proceedings of the 40th International ACM SIGIR conference on research and development in information retrieval. 2017. [slide](https://pdfs.semanticscholar.org/ea73/8439b880ad033ff01602ea52d04b366d0d37.pdf)
- Su, Yuxin, Irwin King, and Michael Lyu. "[Learning to rank using localized geometric mean metrics.](https://arxiv.org/pdf/1705.07563.pdf)" Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2017. [slide](https://www.slideshare.net/YuxinSu/sigir17-learning-to-rank-using-localized-geometric-mean-metrics)
- Dehghani, Mostafa, et al. "[Neural ranking models with weak supervision.](https://arxiv.org/pdf/1704.08803.pdf)" Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2017. [slide](https://mostafadehghani.com/wp-content/uploads/2016/07/SIGIR2017_Presentation.pdf)
- Karmaker Santu, Shubhra Kanti, Parikshit Sondhi, and ChengXiang Zhai. "[On application of learning to rank for e-commerce search.](https://arxiv.org/pdf/1903.04263.pdf)" Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2017.

### 2018

- He, Xiangnan, et al. "[Adversarial personalized ranking for recommendation.](https://arxiv.org/pdf/1808.03908.pdf)" The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018. [code](https://github.com/hexiangnan/adversarial_personalized_ranking)
- Wang, Huazheng, et al. "[Efficient exploration of gradient space for online learning to rank.](https://arxiv.org/pdf/1805.07317.pdf)" The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.
- Dato, Domenico, et al. "[Fast ranking with additive ensembles of oblivious and non-oblivious regression trees.](https://iris.unive.it/retrieve/handle/10278/3692219/191752/paper.pdf)" ACM Transactions on Information Systems (TOIS) 35.2 (2016): 1-31. [slide](https://www.slideshare.net/raffaeleperego/quickscorer-a-fast-algorithm-to-rank-documents-with-additive-ensembles-of-regression-trees)
- Feng, Yue, et al. "[From greedy selection to exploratory decision-making: Diverse ranking with policy-value networks.](http://159.226.40.238/~junxu/publications/SIGIR2018-M2Div.pdf)" The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.
- Ai, Qingyao, et al. "[Learning a deep listwise context model for ranking refinement.](https://arxiv.org/pdf/1804.05936.pdf)" The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018. [code](https://github.com/QingyaoAi/Deep-Listwise-Context-Model-for-Ranking-Refinement)
- Fan, Yixing, et al. "[Modeling diverse relevance patterns in ad-hoc retrieval.](https://arxiv.org/pdf/1805.05737.pdf)" The 41st international ACM SIGIR conference on research & development in information retrieval. 2018. [code](https://github.com/faneshion/HiNT)
- Lucchese, Claudio, et al. "[Selective gradient boosting for effective learning to rank.](https://arca.unive.it/retrieve/handle/10278/3703677/191780/selective-SIGIR2018.pdf)" The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.
- Wu, Liang, et al. "[Turning clicks into purchases: Revenue optimization for product search in e-commerce.](http://www.liangwu.me/files/turning-clicks-purchases.pdf)" The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.

### 2019

- Pasumarthi, Rama Kumar, et al. "[Tf-ranking: Scalable tensorflow library for learning-to-rank.](https://arxiv.org/pdf/1812.00073.pdf)" Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

### 2020

- Khattab, Omar, and Matei Zaharia. "[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf)" Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020. [code](https://github.com/stanford-futuredata/ColBERT)
- Karpukhin, Vladimir, et al. "[Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf)" Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. [code](https://github.com/facebookresearch/DPR)
- Qu, Chen, et al. "[Contextual Re-Ranking with Behavior Aware Transformers.](http://ciir-publications.cs.umass.edu/getpdf.php?id=1383)" Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020.
- MacAvaney, Sean, et al. "[Efficient Document Re-Ranking for Transformers by Precomputing Term Representations.](https://arxiv.org/pdf/2004.14255.pdf)" The 43st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2020. [code](https://github.com/Georgetown-IR-Lab/prettr-neural-ir)
- Zhuang, Honglei, et al. "[Feature transformation for neural ranking models.](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/03d9dbc56c3d1b19a611043a4cb72e227ebba249.pdf)" Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020.
- Lucchese, Claudio, et al. "[Query-level Early Exit for Additive Learning-to-Rank Ensembles.](https://arxiv.org/pdf/2004.14641.pdf)" The 43st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2020.
- Bevendorff, Maik Fröbe1 Janek, et al. "[Sampling Bias Due to Near-Duplicates in Learning to Rank.](https://webis.de/downloads/publications/papers/webis_2020d.pdf)" The 43st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2020. [code](https://github.com/webis-de/SIGIR-20)

### 2021

- Qin, Zhen, et al. "[Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees?](https://openreview.net/pdf?id=Ut1vF_q_vC)" International Conference on Learning Representations (ICLR). 2021.
- Formal, Thibault, Benjamin Piwowarski, and Stéphane Clinchant. "[SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.](https://arxiv.org/pdf/2107.05720.pdf)" Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021. [code](https://github.com/naver/splade)
- Swezey, Robin, et al. "[PiRank: Scalable Learning To Rank via Differentiable Sorting.](https://arxiv.org/pdf/2012.06731.pdf)" Advances in Neural Information Processing Systems (NeurIPS). 2021.
- Oosterhuis, Harrie. "[Computationally Efficient Optimization of Plackett-Luce Ranking Models for Relevance and Fairness.](https://arxiv.org/pdf/2105.05067.pdf)" Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021. (Best Paper Award)
- Hofstätter, Sebastian, et al. "[Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf)" Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021.
- Gao, Luyu, Zhuyun Dai, and Jamie Callan. "[Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline.](https://arxiv.org/pdf/2101.08751.pdf)" European Conference on Information Retrieval (ECIR). 2021.
- MacAvaney, Sean, Franco Maria Nardini, and Raffaele Perego. "[A Systematic Evaluation of Transfer Learning and Pseudo-labeling with BERT-based Ranking Models.](https://arxiv.org/pdf/2106.03699.pdf)" Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021.
- Formal, Thibault, et al. "[A White Box Analysis of ColBERT.](https://arxiv.org/pdf/2101.05405.pdf)" European Conference on Information Retrieval (ECIR). 2021.

### 2022

- Santhanam, Keshav, et al. "[ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.](https://arxiv.org/pdf/2112.01488.pdf)" Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL). 2022. [code](https://github.com/stanford-futuredata/ColBERT)
- Formal, Thibault, et al. "[From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective.](https://arxiv.org/pdf/2205.04733.pdf)" Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022. [code](https://github.com/naver/splade)
- Pobrotyn, Przemysław, et al. "[Learning Neural Ranking Models Online from Implicit User Feedback.](https://arxiv.org/pdf/2204.09118.pdf)" Proceedings of the ACM Web Conference 2022. 2022.
- Khosla, Sopan, and Vinay Setty. "[Risk-Sensitive Deep Neural Learning to Rank.](https://dl.acm.org/doi/abs/10.1145/3477495.3532056)" Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.
- Pasumarthi, Rama Kumar, et al. "[Learning-to-Rank at the Speed of Sampling.](https://dl.acm.org/doi/10.1145/3477495.3531842)" Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.
- Hofstätter, Sebastian, et al. "[Ensemble Distillation for BERT-Based Ranking Models.](https://arxiv.org/pdf/2107.11912.pdf)" Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR). 2021.
- Lassance, Carlos, et al. "[Learned Token Pruning in Contextualized Late Interaction over BERT (ColBERT).](https://arxiv.org/pdf/2203.07785.pdf)" Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.
- Yang, Tao, et al. "[Can clicks be both labels and features? Unbiased Behavior Feature Collection and Uncertainty-aware Learning to Rank.](https://arxiv.org/pdf/2203.11063.pdf)" Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.

### 2023

- Buyl, Maarten, et al. "[RankFormer: Listwise Learning-to-Rank Using Listwide Labels.](https://arxiv.org/pdf/2306.17104.pdf)" Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023.
- Ai, Qingyao, Xuanhui Wang, and Michael Bendersky. "[Metric-agnostic Ranking Optimization.](https://dl.acm.org/doi/10.1145/3539618.3591915)" Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2023.
- Zhao, Haiyuan, et al. "[Unbiased Top-k Learning to Rank with Causal Likelihood Decomposition.](https://dl.acm.org/doi/10.1145/3624918.3625319)" Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP). 2023.

### 2024

- Khramtsova, Ekaterina, et al. "[Leveraging LLMs for Unsupervised Dense Retriever Ranking.](https://arxiv.org/pdf/2402.04853.pdf)" Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2024. (Best Paper Award)
- Borisyuk, Fedor, et al. "[LiRank: Industrial Large Scale Ranking Models at LinkedIn.](https://arxiv.org/pdf/2402.06859.pdf)" Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.
- Huang, Xuyang, et al. "[Unbiased Learning-to-Rank Needs Unconfounded Propensity Estimation.](https://dl.acm.org/doi/10.1145/3626772.3657772)" Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2024.
- Jagerman, Rolf, et al. "[Unbiased Learning to Rank: On Recent Advances and Practical Applications.](https://dl.acm.org/doi/abs/10.1145/3616855.3636451)" Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 2024.
- Liu, Yu-An, et al. "[Multi-granular Adversarial Attacks against Black-box Neural Ranking Models.](https://arxiv.org/pdf/2404.01574.pdf)" Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2024.

### 2025

- Wang, Qingyu, et al. "[From Features to Transformers: Redefining Ranking for Scalable Impact.](https://arxiv.org/abs/2502.03417)" arXiv preprint arXiv:2502.03417. 2025.

## Software

### Classical LTR Libraries

- [Support Vector Machine for Ranking](http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) - SVM^rank implementation
- [Support Vector Machine for Optimizing Mean Average Precision](http://projects.yisongyue.com/svmmap/) - SVM-MAP implementation
- [jforests](https://github.com/yasserg/jforests) - Java implementation of gradient boosted trees for LTR
- [ListNet](https://sourceforge.net/projects/listnet/) - Listwise approach implementation
- [ListMLE](https://sourceforge.net/projects/listmle/) - List Maximum Likelihood Estimation
- [Metric Learning to Rank](https://github.com/bmcfee/mlr) - Python implementation
- [Lerot](https://bitbucket.org/ilps/lerot/src/master/) - Online learning to rank framework
- [xapian-letor](https://github.com/xapian/xapian/tree/master/xapian-letor) - LTR module for Xapian search engine

### Deep Learning & Modern LTR Libraries

- [TensorFlow Ranking](https://github.com/tensorflow/ranking) - Scalable TensorFlow library for LTR
- [LambdaRank Example on LightGBM](https://github.com/Microsoft/LightGBM/tree/master/examples/lambdarank) - LambdaRank with gradient boosting
- [Chainer implementation of RankNet](https://github.com/kzkadc/ranknet) - Neural network approach
- [OpenNIR](https://opennir.net/) - Neural IR research platform
- [metarank](https://www.metarank.ai/) - Modern LTR for e-commerce and recommendations
- [Transformer Rankers](https://github.com/Guzpenha/transformer_rankers) - Library for ranking experiments with transformers

## Dataset

- [LETOR 3.0/4.0](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) - Benchmark datasets for learning to rank research
- [MSLR WEB10K/WEB30K](https://www.microsoft.com/en-us/research/project/mslr/) - Microsoft Learning to Rank datasets
- [TREC QA Track Data](https://trec.nist.gov/data/qamain.html) - Question answering and retrieval datasets
- [Yahoo! Learning to Rank Challenge](https://webscope.sandbox.yahoo.com/) - Large-scale LTR dataset from Yahoo!

## Others

### Tools and Projects

- [QuickRank](https://github.com/hpclab/quickrank) - Fast learning to rank C++ library
- [ExpediaLearningToRank](https://github.com/arifqodari/ExpediaLearningToRank) - LTR application for hotel search

### Japanese Resources (日本語リソース)

- [ランク学習（Learning to Rank） Advent Calendar 2018](https://adventar.org/calendars/3357)
- [DSIRNLP#1 ランキング学習ことはじめ](https://www.slideshare.net/sleepy_yoshi/dsirnlp1)
- [Learning to rank (LTR) とは何か](https://qiita.com/sugiyamath/items/ba08874490e21a9a3ac1)
- [SIGIR2011読み会 3: Learning to Rank](https://www.slideshare.net/sleepy_yoshi/sigir2011-3-learning-to-rank)
- [SIGIR2012勉強会 23: Learning to Rank](https://www.slideshare.net/sleepy_yoshi/sigir2012-23-learning-to-rank)

## Contributing

Contributions are welcome! If you know of any important papers, software, datasets, or resources related to Learning to Rank that are not listed here, please feel free to:

1. Open an issue with the details
2. Submit a pull request with your additions

When adding papers, please:
- Include the full citation with authors, title, and venue
- Add a link to the paper (preferably direct PDF or DOI)
- Place the paper in the appropriate year section
- Follow the existing format

## Star History

If you find this repository useful, please consider giving it a star ⭐

## License

This list is provided as-is for educational and research purposes. All linked papers and resources are copyright of their respective authors and publishers.

