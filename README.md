# Dog vs Cat Image Classifier using PyTorch and ResNet18 

A beginner-friendly deep learning project that uses PyTorch and transfer learning with ResNet18 to classify images as either dogs or cats. This project demonstrates fundamental concepts in computer vision and deep learning while providing practical hands-on experience with a real-world dataset..

## 📋 Project Overview !

This project implements a convolutional neural network (CNN) that can distinguish between images of dogs and cats with high accuracy.
It uses the following:

- **PyTorch**: A popular deep learning framework
- **Transfer Learning**: Leverages pre-trained ResNet18 architecture 
- **Microsoft Dogs vs Cats Dataset**: Contains 25,000 labeled images (12,500 cats and 12,500 dogs)
- **Google Colab**: Cloud-based environment with GPU acceleration

## 📊 Model Performance

After training for 5 epochs, the model achieved:
- **Training accuracy**: ~97.1%
- **Validation accuracy**: ~97.8%
- **Training loss**: 0.072
- **Validation loss**: 0.053

![graph](https://github.com/user-attachments/assets/8acdacaf-5b4f-4228-a8e2-82c618b2f935)

*Training and validation accuracy/loss curves showing model convergence*

## 📷 Prediction Examples

The model successfully classifies new cat and dog images with high confidence-

![belowgraph](https://github.com/user-attachments/assets/ca21443e-1447-4938-8c47-156f9d23d39b)

*Sample predictions showing the model correctly identifying cat and dog images*

## 🚀 How to Run This Project

### Prerequisites
- A Google account (to use Google Colab)
- Internet connection
- No local installations required!

### Steps to run the project 
1. Navigate to the `cat_vs_dog_classifier.ipynb` file in this repo
2. Click on the `.ipynb` file
3. Click on the "Open in Colab" badge at the top of the notebook
   - Or manually open [Google Colab](https://colab.research.google.com/) and upload the notebook file

### Step 2: Setup the Runtime Environment
1. In Google Colab, go to `Runtime` → `Change runtime type`
2. Select `T4 GPU` from the Hardware accelerator dropdown menu
3. Click `Save`

### Step 3: Run the Training Cell
The notebook contains two main cells:

#### Training Cell
1. Click the play button on the first cell or press `Shift+Enter`
2. This will:
   - Mount your Google Drive (to save the model)
   - Download and extract the Microsoft Dogs vs Cats dataset (25,000 images in total)
   - Create training and validation data directories (80/20 split)
   - Define the CNN model architecture using ResNet18
   - Train the model for 5 epochs
   - Plot training/validation loss and accuracy graphs
   - Save the model to your Google Drive

**Note**: Training takes approximately 15-20 minutes on a T4 GPU.

### Step 4: Test the Model
Once training is complete, run the second cell to test your model:
This cell will:
* Load your trained model from Google Drive
* Display an upload button where you can select your own cat/dog images
* Process your uploaded images and display them with predictions
* Show the confidence score for each prediction
* Allow you to test more images by simply re-running the cell

After running the second cell and uploading an image, you'll see results like this:

![image](https://github.com/user-attachments/assets/d51ecf1b-8f65-47de-9fe5-c644fe78563f)
![image](https://github.com/user-attachments/assets/ba90e6a6-aa1f-449f-8854-aeda2486af6c)

*The model correctly identifies the images as a cat with 99.63% confidence and a dog with 100.0% confidence.*

You can upload multiple images at once to see how the model performs on different cat and dog photos.



## 🧠 Technical Details

### Dataset
- **Microsoft Dogs vs Cats dataset**: Originally from Kaggle competition
- Contains 25,000 images:
  - 12,500 cat images
  - 12,500 dog images
- Various breeds, poses, and backgrounds
- Images are resized to 224x224 during preprocessing
- Dataset is split into:
  - 80% training data (~20,000 images)
  - 20% validation data (~5,000 images)

### Architecture -
- **Base Model**: ResNet18 pre-trained on ImageNet
- **Modifications**: Custom classifier head for binary classification
- **Input Size**: 224x224 RGB images
- **Optimization**: Adam optimizer with learning rate of 0.001
- **Loss Function**: Cross-Entropy Loss
- **Normalization**: Using ImageNet mean and standard deviation

## 🔍 What You'll Learn

This beginner-friendly project demonstrates:
- How to prepare image data for deep learning
- Transfer learning with pre-trained models
- Training deep learning models with PyTorch
- Model evaluation and visualization techniques
- Practical deployment for inference

## ⭐ Star this project if you found it useful :)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
