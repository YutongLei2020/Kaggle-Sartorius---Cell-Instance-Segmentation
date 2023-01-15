# Kaggle-Sartorius---Cell-Instance-Segmentation
Kaggle Competition link: https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation
### 1. Introduction and Problem Statement
This problem aims to aid Alzheimer’s research by addressing the issue of neuronal cells being difficult to individually segment. In Alzheimer's research studies, light microscopy can be used to evaluate whether treatments are effective. However, the segmenting process in the analysis of the neuronal cells is a challenging and time consuming task. In this project, we attempt to address this challenge by providing a program that has been trained on phase contrast images to segment neuronal cells when provided a new image of cells. This project will use U-net to delineate objects in microscopy biological images that represent neurological disorders, with evaluation using mean average precision.
### 2. Related Work:
Many methods have been used for medical image segmentation. Some of the most common approaches to image segmentation include artificial neural networks, thresholding and classifiers. Artificial neural networks pass information through various “neurons” that are assembled into layers with corresponding weights (determined by training data) to achieve learning.  For example, a U-net model has been developed for iris recognition (Lian, Sheng, et al.) that first generates a bounding box for the iris region, and an attention mask that serves as a weighted function for segmentation along with feature maps. Thresholding is where partitions are made on the image depending on image intensity in order to segment different elements of the image. Threshold image intensities are used to group and segment pixels. Classifiers use pattern recognition techniques to partition certain elements of the image and require reference or training data (supervised learning).
In this project, we systematically evaluated the performance of U-net models in their application for medical image segmentation. Specifically, we are evaluating how the combination of different layers, such as Dropout, MaxPooling, and BatchNormalization, in a U-net affect the generation of masks for segmenting neuronal cells. 
### 3. Data Sets:
The training data set consists of 606 images in png format. Each image will have 9 classification labels, including an object ID, a run length for the neuronal cell, image width, image height, cell line, time plate was created, date sample was created, sample ID, and time since the first sample image was taken. Figure 1, 2 and 3 shows sample images of the three cell types, mask color and mask.

Link to the Kaggle Data description of the Sartorius - Cell Instance Segmentation:
https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data

The test images are in PNG format. Only a few test set images are available for download from Kaggle; the remainder can only be accessed by your notebooks when you submit.

The testing images also have a related test.csv that contains some information about the test images, including ids, annotations, width, height, cell type, plate time, sample date, sample id, and elapsed timedelta. Since most of the information is irrelevant to the image segmentation, we decided to only extract annotations of unique ids in the .csv file. 
### 4. Description of Technical Approach
The method we are using to develop a model is U-net. This is a convolutional neural network that uses successive layers to generate a high resolution output. High resolution is attained through a large number of features. We are using this method because it works well for biomedical image segmentation, which is the problem we are addressing in this project

U-net takes an image as input and it starts with contraction; it reduces the resolution as well as increases the channel size of each layer. U-net starts the expansion process when it reaches the bottleneck layer; U-net increases the resolution and decreases the channel size, it also concatenates the results of the contraction layers with its result.

We used tensorflow.keras to implement our model and to evaluate our model, we are using three Keras metrics: SparseCategoricalAccuracy(), BinaryCrossentropy(), and Mean().
### 5. Software

Tables | functionality | source | Link to the public source
------ | -------------- | ------------- | --------------------------
Demonstration of cell types | Shows the visualization of different cell types. | The helper functions are from an outside source. Other codes are written by us. | (https://www.kaggle.com/code/danduong/segment-using-unet) 
Data extraction | Read the image and annotations into the form of arrays. | We wrote this part. |
Split image | It split the image and annotations into smaller pieces to increase sample size as well as making the image easier to process. | This part is from a public source | https://www.kaggle.com/code/israrahmed919/unet-on-splitting-data
U-net | The model that processes the dataset and makes predictions. | The model is from a public source. We made modifications to the model. | https://www.kaggle.com/code/israrahmed919/unet-on-splitting-data
Train | Train the model and visualize the process of training. | The training process is a combination of public source and our own code. | https://www.kaggle.com/code/israrahmed919/unet-on-splitting-data
Prediction | Make predictions of the test data, and show the visualization of the prediction. | We wrote this part.



### 6. Experiments and Evaluation 
The input of a neural network model should be in the form of numerical numbers, so we used imageio.imread to read the input images as arrays in size of (520,704).
Also, 606 images is not a large number for the train dataset, and we decided to separate the image into smaller images in order to increase the sample size. In order to make the image easier for the model to process, we separate the images into square images of 128*128, and each original image can be separate to 20 128*128 images. Since 704%128 = 64, so the remaining 64 * 520 pixels would be dropped.

Figure 5: Image splitting to increase the training dataset sample size
We started with an existing solution to this problem. The model is a full layers U-net with 10 epochs of training. It performed well on both training and validation sets. It was stabled at a precision around 0.88 before the first epoch of training ended. The training losses and the validation losses are different; the training losses tend to be unstable, it can be from 0.1 to 1.5.
However, the validation losses are much more stable than the training losses; it was stabled to 0.3 after three epochs of training. The accuracy and losses barely change after three epochs, so we figured too many epochs is not necessary, and we reduced the number of epochs to three. It performed similarly with fewer epochs.
We tried simplified versions of U-net with only fewer layers, but it still contains all the required elements of a U-net, including a contraction layer, a bottleneck layer, and an expansion layer. Although the training accuracy is similar to a full layers U-net model, when we visualize the segmentation of the test dataset, we can see the difference.

Figure 6: Comparison of segmentation masks using a 3-layer U-net vs. a 9-layer U-net
We can see that the U-net model with more layers detects more details in the image.
When we retrain the three layers version of U-net several times, the visualization of segmentation on the test dataset looks just as good as a full layers U-net model. The three layers model had a much faster runtime compared to the full layers version, even with more epochs than the full layers model, it’s still faster.

Figure 7: Comparison of segmentation masks using a 3-layer U-net with 9 Epochs of training vs. a 9-layer U-net with 3 Epochs
We also added a dropout layer and a batch normalization layer to the model to improve the performance.
6. Discussion and Conclusion 
We learned a lot about the optimization of U-nets from this project. The first problem we encountered was the low number of training data. Since this is a segmentation problem not a classification problem, it won’t affect anything if we split the images, and we found a way to split each image into multiple images with smaller area in order to increase the sample size.
From the results we obtained, we can say that it reaches our expectation; it can segment the image successfully. For the parameters of the model, it did not go as we expected. We were expecting a full layers U-net would perform much better than a U-net with a lower number of Encoding and Decoding layers. Instead, a three-layer U-net with more epochs can perform similarly to a full-layer U-net with less runtime.
In the future, we think it could be interesting to further explore how image splitting affects image segmentation accuracy since small sample sizes can be challenging to work with.  

### References / Work Cited

Lian, Sheng, et al. “Attention Guided U-Net for Accurate Iris Segmentation.” Journal of Visual Communication and Image Representation, vol. 56, Oct. 2018, pp. 296–304. ScienceDirect, https://doi.org/10.1016/j.jvcir.2018.10.001.
Pham, Dzung L., et al. “Current Methods in Medical Image Segmentation.” Annual Review of Biomedical Engineering, vol. 2, no. 1, Aug. 2000, pp. 315–37. DOI.org (Crossref), https://doi.org/10.1146/annurev.bioeng.2.1.315.
