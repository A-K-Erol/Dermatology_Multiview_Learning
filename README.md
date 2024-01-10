# Dermatology_Multiview_Learning

This project explores three AI / Machine Learning techniques on a multiview dermatology dataset from the University of California, Irvine, 
available at https://archive.ics.uci.edu/dataset/33/dermatology. This is one of several datasets used to benchmark multiview algorithms. 
Data entries with missing rows were pruned from the dataset, because they accounted for <1.5% of entries.

I obtained the following testing accuracy:
  1. Support vector machine: 99.4% accuracy
  2. Neural Network: 97.1% accuracy
  3. Multi-view K Nearest Neighbor: 94.3% accuracy
  4. XGBoost Decision Tree: 98.6% accuracy

Data Preparation:
  1. Feature 34 was normalized to be on the same scale as features 1 through 33.
  2. Dimensionality Reduction was performed via Principal Component Analysis, with term count = 10, 16, 30 ascertained by validation. Linear PCA had good results, though there appeared to be slight improvements with Kernel PCA on different Kernels.
  3. Model-specific cross-validation libraries were used to partition training/testing data for cross-validation with k = 6. Error metrics computed included logloss, RMSE, and precision.

Details:
  1. SVM: I used the sklearn library to implement the algorithm. Tested with both a one-vs-one and one-vs-rest approach for multiple classification, with nearly identical results.
  2. Neural network: I used the pytorch library and pandas to process the data; testing was performed with several neural networks, with the best results for simple and small linear network models. Epoch count of 200 and learning rate of .001 were used. 
  3. KNN: I coded this from scratch with usage solely of libraries for initial data processing. The results were worse than those obtained in research studies using algorithms with comparable approaches, such as Kiyak (2021). The voting mechanism is somewhat poor, and better results could likely be obtained if all features were considered at once, since majority voting biases the first view if the classifiers disagree.
  4. XGBoost: Lots of experimentation was done here, but baseline parameters seemed to work relatively optimally. There was a strong tendency to overfit the data which hurt cross-validation results, and reducing complexity with the number of parallel trees yielded improvements to validation results.

