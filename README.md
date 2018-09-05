# PCA-vs-Neural-Network-Dimensionaity-Reduction

This repository is comparing dimensionality reduction techniques: Neural Networks and Priciple Component Anaylsis. It is testing to see whether there are any differences when the two techniques try to cluster rap music vs classical music. 

The file cutsongs.py cuts entire songs into five second clips. 

The file NN_Dimension_Reduction.py reads in those clips and trains an autoencodder neural net using them as inputs. It then records the middle hidden layer (3 nodes wide) for each song passed through the trained neural net and graphs those points. In the end it saves that graph. 

The code for doing PCA on the songs has been lost although it was not that difficult. 

The report is included in the reposity as well detailing the results.
