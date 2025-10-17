# 7 Wonder Places Classification with CNN

## Project Overview
This project aims to evaluate and compare the performance of four Convolutional Neural Network (CNN) pre-trained models—VGG16, ResNet50V2, Xception, and InceptionV3—in classifying images of the New 7 Wonders of the World. By leveraging transfer learning, we investigate how pre-trained models can enhance classification accuracy on custom datasets.

## New 7 Wonders of the World
The wonders considered in this project are:

1. Great Wall of China
2. Petra (Jordan)
3. Christ the Redeemer (Brazil)
4. Machu Picchu (Peru)
5. Chichen Itza (Mexico)
6. Colosseum (Italy)
7. Taj Mahal (India)

## Dataset
The dataset consists of images representing each of the seven wonders, which were previously selected through a global voting campaign. Images were preprocessed to ensure uniformity and improve model performance.

![Great Wall of China](https://github.com/user-attachments/assets/50cab860-9c25-488a-9669-d8496bd85a20)
*Great Wall of China*

![Petra](https://github.com/user-attachments/assets/e7b3da22-9cc7-4211-ad75-b156c42f64bb)
*Petra, Jordan*

![Christ the Redeemer](https://github.com/user-attachments/assets/19e06ebe-a6a1-49ce-8419-a8c72556e9a9)
*Christ the Redeemer, Brazil*

![Machu Picchu](https://github.com/user-attachments/assets/2d59ed7e-7616-44a1-ab65-99fa203e07df)
*Machu Picchu, Peru*

![Chichen Itza](https://github.com/user-attachments/assets/ccd14e66-6334-44c4-8eda-c5a3851bf8e9)
*Chichen Itza, Mexico*

![Colosseum](https://github.com/user-attachments/assets/69e994b6-ca0c-4d86-a785-ac85cfa6e3e0)
*Colosseum, Italy*

![Taj Mahal](https://github.com/user-attachments/assets/cbbe599c-3383-4eb1-bb8b-51c2af7aefc3)
*Taj Mahal, India*

## Methodology
### Data Preprocessing
To ensure high performance of the models, the following preprocessing steps were undertaken:
- Image resizing
- Normalization
- Data augmentation techniques to create variations

### Model Architecture
Four models were utilized for this project:
- **VGG16**
- **ResNet50V2**
- **Xception**
- **InceptionV3**

### Training Strategy
Models were trained using:
1. **Transfer Learning:** Fine-tuning on the New 7 Wonders dataset.
2. **Feature Extraction:** Using the convolutional base to extract features.

### Visualization Technique: Grad-CAM
Grad-CAM was applied to visualize the regions of images that influenced model predictions, thereby improving interpretability.

## Results
Models were evaluated based on accuracy, precision, recall, F1-score, and confusion matrices. Grad-CAM visualizations provided insights into model decision-making processes.

## Conclusion
The project demonstrated the power of transfer learning for image classification tasks, with specific models outperforming others on customized datasets. Insights gained from Grad-CAM highlighted crucial image features contributing to model predictions.

## Dependencies
To run this project, you will need the following Python packages:
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV
- scikit-learn

## Installation
```bash
pip install -r requirements.txt
