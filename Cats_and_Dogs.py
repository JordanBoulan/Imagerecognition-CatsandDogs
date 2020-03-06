
# coding: utf-8

# I wanted to learn how machine learning is used to classify images (Image recognition).
#  I was browsing Kaggle's past competitions and I found Dogs Vs Cats: Image Classification Competition (https://www.kaggle.com/c/dogs-vs-cats. 
# Here one needs to classify whether image contain either a dog or a cat). Google search helped me to get started. Here are some of the references that I found quite useful: Yhat's Image Classification in Python and SciKit-image Tutorial. 
# Data is available at https://www.kaggle.com/c/dogs-vs-cats. Here I am using first 501 dog images and first 501 cat images from train data folder. For testing I selected first 100 images from test data folder and manually labeled image for verifying.



# Import necessary libraries
import pandas as pd
import numpy as np
from skimage import io 
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

train_directory = "./data/train/"
test_directory = "./data/test/"


# In[5]:

# Define a function to return a list containing the names of the files in a directory given by path
def images(image_directory):
    return [image_directory+image for image in os.listdir(image_directory)]

images(train_directory)


# In[6]:

##In training directory, image filename indicates image label as cat or dog: Need to extract labels
## Extracting training image labels
train_image_names = images(train_directory)


# In[7]:

# Function to extract labels
def extract_labels(file_names):
    '''Create labels from file names: Cat = 0 and Dog = 1'''
    
    # Create empty vector of length = no. of files, filled with zeros 
    n = len(file_names)
    y = np.zeros(n, dtype = np.int32)
    
    # Enumerate gives index
    for i, filename in enumerate(file_names):
        
        # If 'cat' string is in file name assign '0'
        if 'cat' in str(filename):
            y[i] = 0
        else:
            y[i] = 1
    return y

extract_labels(train_image_names)


# In[8]:

# Save labels
y = extract_labels(train_image_names)

# Save labels: np.save(file or string, array)
np.save('y', y)

# Images in test directory
images(test_directory)


# In[9]:

## View image: Dog
# from skimage import io # (imported earlier)
temp = io.imread('./data/train/dog.20.jpg') 
plt.imshow(temp)


# In[10]:

## View image: Cat
# from skimage import io # (imported earlier)
temp = io.imread('./data/train/cat.4.jpg') 
plt.imshow(temp)


# In[11]:

from PIL import Image


# In[16]:

# Using folder sort I found that Images are of different sizes: (max size = cat.835.jpg, min size = cat.4821.jpg). Need a standard size for analysis

# Get size of images (Ref: stackoverflow)
image_size = [ ]

for i in train_image_names: # images(file_directory)
    im = Image.open(i)
    image_size.append(im.size) # A list with tuples: [(x, y), …] 

# Get mean of image size (Ref: stackoverflow)
[sum(y) / len(y) for y in zip(*image_size)]


# In[17]:

## Transforming the image: Standard size = (400, 350)

## Transforming the image
# Set up a standard image size based on approximate mean size

STANDARD_SIZE = (400, 350)

## Code below copied from: Yhat's Image Classification in Python: http://blog.yhathq.com/posts/image-classification-in-Python.html

# Function to read image, change image size and transform image to matrix
def img_to_matrix(filename, verbose=False):
        
    '''
    takes a filename and turns it into a numpy array of RGB pixels
    '''
    img = Image.open(filename)
    # img = Image.fromarray(filename)
    if verbose == True:
        print "Changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

# Function to flatten numpy array
def flatten_image(img):
    
    '''
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    '''
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


# In[18]:

## Prepare training data
data = []
for i in images(train_directory):
    img = img_to_matrix(i)
    img = flatten_image(img)
    data.append(img)
    
data = np.array(data)
data.shape


# In[19]:

data[1].shape


# Total 420,000 features per image. 420,000 features is a lot to deal with for many algorithms, so the number of dimensions should be reduced somehow. For this we can use an unsupervised learning technique called PCA to derive a smaller number of features from the raw pixel data. Principal Component Analysis (PCA):  to identify patterns to reduce dimensions of the dataset with minimum loss of information.

# In[20]:

# Import PCA
from sklearn.decomposition import PCA

# PCA on training data
pca = PCA(n_components = 2)
X = pca.fit_transform(data)
X.size


# In[21]:

print X[:, 0].size
print X[:, 1].size




# Create a dataframe
np.sum(pca.explained_variance_ratio_)


# In[22]:

# Here 2-Dimension PCA, captures 64.6% of the variation

## Prepare testing data: PCA
test_images = images(test_directory)

test = [ ]
for i in test_images:
    img = img_to_matrix(i)
    img = flatten_image(img)
    test.append(img)

test = np.array(test)
test.shape


# In[23]:

# Transforming test data
testX = pca.fit_transform(test)
testX.shape[1]


# In[24]:

## Logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression( )
logreg = clf.fit(X, y)


# In[25]:

# Predict using Logistic Regression





## Logistic Regression: Accuracy

# Load 'Actual' labels for test data
actual = pd.read_csv('ActualLabels.csv')
actual['Labels'].head( ) 




logreg_accuracy = np.where(y_predict_logreg == actual['Labels'], 1, 0).sum()/float(len(actual))


# 54% of the images were correctly classified using logistic regression (2D PCA)


# In[23]:

## KNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X, y)


# In[24]:

# Predict using KNN classifier
y_predict_knn = knn.predict(testX)


# In[25]:

## KNN: Accuracy
knn_accuracy = np.where(y_predict_knn == actual['Labels'], 1, 0).sum()/float(len(actual))







