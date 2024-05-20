# Dog Breed Classification using CNN Model on Stanford Dogs Dataset
## Description
The <a href= "http://vision.stanford.edu/aditya86/ImageNetDogs/">Stanford Dogs Dataset</a> contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.

I have used the NasNetLarge CNN Model, which is pre-trained on the ImageNet dataset for classification. Data augementation has been used for making the model generalize better and also to avoid overfitting. The model achieved an accuracy of 93% on validation set, which is decent for this dataset.

## Getting Started

### Pre-Requisites
If you need to use a lab computer, you need to install a virtual environment to install all the requirement in the requirements.txt
### Installation
**Dependencies:**
```
# With Tensorflow CPU
pip install -r requirements.txt

# With Tensorflow GPU
pip install -r requirements-gpu.txt
```
## Dataset
Contents of the dataset:
- Number of categories: 120
- Number of images: 20,580
- Annotations: Class labels, Bounding boxes

The dataset can be downloaded from <a href= "http://vision.stanford.edu/aditya86/ImageNetDogs/">here.</a>

In my project, I only used 9 categories, so put the 9 breeds' images and annotation into the corresponding folders(Images, Annotation).
Put the test_data.mat into the test folder.


## Approach
### Prepare the data
Run the PrepareTheData.py to get the label, the allDogsImages.npy and the allDogsLabels.npy

![Augmented Image](/images/augmented_image.png)

### Model Training
Run the NasNetLargeModel.py to get the trained model

### Training Results
Run the TestTheModel.py to get the results of the 9 pics
Run the TestTheMetrics.py to get the results of the performance

## References
- The original data source is found on http://vision.stanford.edu/aditya86/ImageNetDogs/ and contains additional information on the train/test splits and baseline results.
- Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.  <a href="http://people.csail.mit.edu/khosla/papers/fgvc2011.pdf">[pdf]</a> <a href="http://vision.stanford.edu/documents/KhoslaJayadevaprakashYaoFeiFei_FGVC2011.pdf">[poster]</a> <a href="http://vision.stanford.edu/bibTex/KhoslaJayadevaprakashYaoFeiFei_FGVC2011.bib">[BibTex]</a>