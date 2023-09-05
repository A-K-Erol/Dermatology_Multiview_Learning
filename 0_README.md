# Dermatology_Multiview_Learning

This project explores three AI / Machine Learning techniques on a multiview dermatology dataset from the University of California, Irvine, 
available at https://archive.ics.uci.edu/dataset/33/dermatology. This is one of several datasets used to benchmark multiview algorithms. 
Data entries with missing rows were pruned from the dataset, because they accounted for <1.5% of entries.

I obtained the following testing accuracy:
  1. Support vector machine: 74.2% accuracy
  2. Neural Network: 93.8% accuracy
  3. Multi-view K Nearest Neighbor: 74% accuracy

Details:
  1. SVM: I used the sklearn library to implement the algorithm. Tested with both a one-vs-one and one-vs-rest approach for
     multiple classification, with nearly identical results.
  2. Neural network: I used the pytorch library and pandas to process the data; testing was performed with several neural networks, with the best results
      for simple and small linear network models. Epoch count of 200 and learning rate of .001 were used.
  3. KNN: I coded this from scratch with usage solely of libraries for initial data processing. The results were worse than those obtained in research
    studies using algorithms with comparable approaches, such as Kiyak (2021). The voting mechanism is particularly poor, and better results could likely
    be obtained if all features were considered at once, since majority voting biases the first view if the classifiers disagree.

