Project Name: 7 Wonder Places Classification with CNN

## ✨ Highlight

1. Based on pre-trained model, the highest average accuracy rate is 94.6% achieved by 🥇 ResNet50V2.

2. The best model after fine-tuning is 🏆 InceptionV3, with an average accuracy of 97.47%, improving from 93.3% (no fine-tuning).

3. Both InceptionV3 and VGG16 show significant improvements of +4.17% and +11.26% in test accuracy, respectively.
Meanwhile, ResNet50V2 improves by +2.77%, and NASNetMobile shows the largest jump of +71.0%, indicating strong adaptation after fine-tuning.

4. Overall, the models effectively learned distinctive visual features of the Seven Wonders of the World, enabling accurate classification through feature extraction and fine-tuned representation learning.

## 📑 Table of Contents
1. [Introduction 🎯](#1-introduction)
2. [Dataset Description 📜](#2-Dataset-Description)
3. [Data Sources 📘](#3-Data-Preparation-and-Pre-Processing)
4. [Model architecture 🧠](#4-Model-architecture)
5. [Results 📊](#5-Training-method)
6. [Experimental Results 💬](#6-Experimental-Results)
7. [Discussion and Conclusion 🧾](#7-Discussion-and-Conclusion)
8. [References 🌐](#8-References)
9. [🎥 Member, Contribution and Responsibility](#Member,-Contribution-and-Responsibility)
10. [End credit ](#-End-credit)


## 1. Introduction
This project aims to evaluate and compare the performance of four CNN pre-trained models — VGG16, ResNet50V2, Xception, and InceptionV3 — originally trained on the ImageNet dataset, by fine-tuning them to classify the New 7 Wonders of the World, which include: 
```
1. Great Wall of China 
2. Petra (Jordan) 
3. Christ the Redeemer (Brazil) 
4. Machu Picchu (Peru) 
5. Chichen Itza (Mexico) 
6. Colosseum (Italy) 
7. Taj Mahal (India)
```
 
The main objective is to compare the performance of these four models with and without transfer learning (fine-tuning) to analyze how pre-trained knowledge on large-scale datasets like ImageNet can enhance classification accuracy when adapted to a custom image dataset that the models have never seen before.
 
Finally, the Grad-CAM (Gradient-weighted Class Activation Mapping) technique is applied to interpret the model’s decision-making process and visualize which image regions most influence the classification. This provides better understanding of
 
Overall, this study aims to explore the effectiveness of transfer learning and CNN visualization techniques in real-world image classification tasks, contributing to both academic learning and practical implementation in computer vision.
 
## 2. Dataset Description
The dataset used in this project consists of images representing the New 7 Wonders of the World, which are world-famous landmarks selected through a global voting campaign. Each class corresponds to one of the seven wonders, as listed below:
 
**Great Wall of China — China** 

<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/50cab860-9c25-488a-9669-d8496bd85a20" />

 
**Petra — Jordan** 

<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/e7b3da22-9cc7-4211-ad75-b156c42f64bb" />
 
**Christ the Redeemer — Brazil** 

<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/19e06ebe-a6a1-49ce-8419-a8c72556e9a9" />
 
**Machu Picchu — Peru** 

<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/2d59ed7e-7616-44a1-ab65-99fa203e07df" />
 
**Chichen Itza — Mexico** 

<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/ccd14e66-6334-44c4-8eda-c5a3851bf8e9" />
 
**Colosseum — Italy** 

<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/69e994b6-ca0c-4d86-a785-ac85cfa6e3e0" />
 
**Taj Mahal - India**

<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/cbbe599c-3383-4eb1-bb8b-51c2af7aefc3" />

### 2.1 Data Sources
The images were collected from open-source image repositories such as Unsplash, Google Images using keyword-based web scraping. Each category contains a diverse set of photos that vary in lighting conditions, camera angles, weather, and distance, ensuring that the model can generalize across real-world scenarios.

### 2.2 Imbalance dataset
<!-- Check amount of images per classes -->
<p align="center">
  <img width="500" height="292" alt="image" src="https://github.com/user-attachments/assets/aad620ea-f0ca-469b-89fa-f961b6313f29" />
  <br>
  <b>Figure 1.</b> Number of images per class (Bar chart overview)
  <br><br>
  <img width="1790" height="489" alt="image" src="https://github.com/user-attachments/assets/036566db-ab2a-4bfc-a65b-45f846145fb3" />
  <br>
  <b>Figure 2.</b> Detailed image distribution across all classes
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/f560781b-eab6-4623-b771-4f6e9e6967f5" alt="Train dataset" width="400"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/e859de4e-a403-431f-938d-1561c24f89a4" alt="Validation dataset" width="400"/>
  <br>
  <b>Figure 3.</b> Image quality samples - <i>Training</i> and <i>Validation</i> datasets
</p>

<p align="center" style="margin-top: 15px;">
  <img src="https://github.com/user-attachments/assets/581e144b-e19c-4e63-b10b-469f9b396277" alt="Test dataset" width="1000"/>
  <br>
  <b>Figure 5.</b> Image quality samples - <i>Test dataset</i>
</p>


## 3. Data Preparation and Pre-Processing    
Before training, all images were preprocessed as follows:  
- Resizing: Each image was resized to 224×224 pixels to fit the input shape of pre-trained CNN models.
  

- Augmentation technique were applied to increase dataset diversity and reduce overfitting.  

<table align="center">
  <tr>
    <td colspan="2" align="center">
       <img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/a1db14a9-bde7-4037-8c56-23a089ac8a68">
        <b>Figure 6.</b> 
    </td>
  </tr>
  <tr>
    <td><img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/b3641839-3c69-4de0-b58e-f08968411abd"></td>
    <td><img width="800" height="800" alt="undefined" src="https://github.com/user-attachments/assets/25509b70-f372-4295-8e21-7fe633a4ad9e" /></td>
  </tr>
</table>

- Normalization: Pixel values were scaled to a range of [0, 1] for faster convergence during training
- Data Splitting (Train/ Validation / Test)
  The dataset was then split into three subsets:  
    - Training set: 70% of total (images = 210  images/class)   
    - Validation set: 15% of total (images = 30 images/class)  
    - Test set: 15% of total images (images = 30 images/class)
 
  This dataset serves as a custom, domain-specific dataset that differs significantly from the ImageNet dataset, allowing us to evaluate how well pre-trained CNN architectures can adapt to unseen image domains through fine-tuning.

## 4. Model architecture

### Pre-training Models 
In this experiment, we have selected 4 Pre-training Models for fine-tuning with IMAGENET dataset as weight Pre-training Models Information.

<p align="center">
  <img width="1104" height="184" alt="image" src="https://github.com/user-attachments/assets/a9f47852-3027-4043-89d6-8cb708847464"/>
  <br>
</p>

### Baseline Evaluation of Pre-trained CNN Models on the New 7 Wonders Dataset

<p align="center">
  <!-- แถวที่ 1 -->
  <img src="https://github.com/user-attachments/assets/b73d24e1-0cfc-415b-8193-a0914d16cbcc" width="30%">
  <img src="https://github.com/user-attachments/assets/644986e1-cc7b-4816-93d7-cf7ca94fde9d" width="30%">
  <img src="https://github.com/user-attachments/assets/f3328dd0-b5d7-4ba3-a583-c0d33dfc5139" width="30%">
  <br>
  <!-- แถวที่ 2 -->
  <img src="https://github.com/user-attachments/assets/37c33c26-553a-4f5a-a1fc-aadea198d7e9" width="30%">
  <img src="https://github.com/user-attachments/assets/e48d4163-9c01-4ceb-b391-66a77cefb653" width="30%">
  <img src="https://github.com/user-attachments/assets/9ce2bdd2-f245-484b-90ce-363c49469406" width="30%">
</p>


### 4.1 VGG16 

- Network Architecture of Pre-training model without fine-tuning vs with Fine tuning - **VGG16**
<p align="center">
  <img width="1125" height="417" alt="image" src="https://github.com/user-attachments/assets/035a117a-0a98-4ff8-80d9-ffd196fd815f" />
  <br>
</p>
  
-  Network Architecture of Pre-training without Fine-tuning - **VGG16**
<p align="center">
  <img width="327" height="594" alt="image" src="https://github.com/user-attachments/assets/69502149-1910-4a16-8832-b341602d0b12" />
  <br>
</p>
  
-  Network Architecture of Pre-training with Fine-tuning (The Best Performance Model)- **VGG16**
<p align="center">
  <img width="327" height="939" alt="image" src="https://github.com/user-attachments/assets/a09fda50-7e53-486e-a92c-df8fd795d52f"/>
  <br>
</p>

### 4.2 RestNet50V2

- Network Architecture of Pre-training model without fine-tuning vs with Fine tuning - **RestNet50V2**


<p align="center">
  <img width="1209" height="385" alt="image" src="https://github.com/user-attachments/assets/5a1b98e7-a9a1-4edb-bd4a-01310b6eb118"  />
  <br>
</p>
  
-  Network Architecture of Pre-training without Fine-tuning - **RestNet50V2**
<p align="center">
  <img width="426" height="858" alt="image" src="https://github.com/user-attachments/assets/f3418669-1476-4d87-be33-fa0a0d3c20c5" />
  <br>
</p>
  
-  Network Architecture of Pre-training with Fine-tuning (The Best Performance Model)- **RestNet50V2**
<br>
<table align="center">
  <tr>
    <td><img width="526" height="1035" alt="image" src="https://github.com/user-attachments/assets/5ba8432b-b086-4a1c-a686-e30f67b46194"></td>
    <td><img width="526" height="1035" alt="image" src="https://github.com/user-attachments/assets/bef79504-2ef6-47f4-9006-adc5a50f07fd"></td>
  </tr>
</table>


### 4.3 Inception V3

- Network Architecture of Pre-training model without fine-tuning vs with Fine tuning - **InceptionV3**
<p align="center">
  <img width="1342" height="412" alt="image" src="https://github.com/user-attachments/assets/1b7ff7a1-39d2-41a1-b10c-360731c275c0" />
  <br>
</p>

- Network Architecture of Pre-training without Fine-tuning - **InceptionV3**
<p align="center">
  <img width="304" height="676" alt="image" src="https://github.com/user-attachments/assets/5a90830c-5d8e-4f7a-abf1-4e18f27ef194" />
  <br>
</p>

- Network Architecture of Pre-training with Fine-tuning (The Best Performance Model)- **InceptionV3** - **Unfrozen last 40 layers**
<img width="761" height="285" alt="image" src="https://github.com/user-attachments/assets/d48fb231-5403-48f5-84f1-ed7ab5be2b61" />

### 4.4 NASNetMobile

- Network Architecture of Pre-training model without fine-tuning vs with Fine tuning - **NASNetMobile**
<p align="center">
  <img width="1207" height="512" alt="image" src="https://github.com/user-attachments/assets/e70848f2-6eb1-44cb-be20-92242f2332c0" />
  <br>
</p>

- Network Architecture of Pre-training with and without out Fine-tuning - **NASNetMobile**

<p align="center">
  <img width="313" height="559" alt="image" src="https://github.com/user-attachments/assets/997ad048-9ee7-4ee5-9353-236bc34576c0" />
  <br>
</p>

-  Network Architecture of Pre-training with Fine-tuning (The Best Performance Model)- **NASNetMobile**
<p align="center">
  <img width="1465" height="518" alt="image" src="https://github.com/user-attachments/assets/d67c7e42-a2f9-42ba-a7eb-79a897efdab2" />
  <br>
</p>

## 5. Training method
*Training Configuration*  

The table below summarizes the key hyperparameter settings used for training all CNN models in both cases — before and after fine-tuning.
All models were trained using the same optimizer and loss function to ensure consistent evaluation conditions. 

We Conducted hyperparameter tuning to identify the optimal configuration that maximizes model performance. (By using Keras Tuner) 
<p align="center">
  <img width="1161" height="325" alt="image" src="https://github.com/user-attachments/assets/621a3554-1370-4e95-acf1-f564630aa27c" />
  <br>
</p>

 
## 6. Experimental Results
---
### 6.1 Training and Validation Accuracy/Loss Analysis

- VGG16
<p align="center">
 <img width="1053" height="348" alt="image" src="https://github.com/user-attachments/assets/51538c4e-ec29-42af-9780-311640213893" />
  <br>
</p>


- RestNet50V2
<p align="center">
 <img width="1053" height="348" alt="image" src="https://github.com/user-attachments/assets/ad275a9b-a565-4ee9-a0b0-774ff7a0a30a" />
  <br>
</p>

- InceptionV3
<p align="center">
 <img width="1053" height="348" alt="image" src="https://github.com/user-attachments/assets/f93f51a6-aa0e-4ce9-ba7d-a7ffd72fd5f0" />
  <br>
</p>

- NASNetMobile
<p align="center">
 <img width="1053" height="348" alt="image" src="https://github.com/user-attachments/assets/2aa6381f-4c41-46a2-9a21-bccc447cbf97" />
  <br>
</p>

### 6.2 Test Accuracy/Loss Analysis
We pre-train the model with initial random weights in the first round and more 2 rounds by another random seed to calculate mean±SD of accuracy and loss on test set as the average of the model performance In each round, accuracy and loss of test sets are not significantly different. That proves the model is good fit.  
<p align="center">
 <img width="1064" height="256" alt="image" src="https://github.com/user-attachments/assets/295a7d3a-1410-41a9-9a64-3d6a6964fbf5" />
  <br>
</p>


### 6.3 Evaluation Metric on Test Set  
<p align="center">
 <img width="800" height="711" alt="image" src="https://github.com/user-attachments/assets/054bfdc9-bf72-401f-9340-30c980cb0b14" />
  <br>
</p>
The confusion matrices show that ResNet50V2 delivers the best performance overall, with the clearest diagonal and very few misclassifications. InceptionV3 and VGG16 also perform well, with only minor confusion between a few landmark classes. In contrast, NASNetMobile shows more frequent misclassifications, especially for Taj Mahal, Rome Colosseum, and Machu Picchu. Overall, ResNet50V2 appears to be the most reliable model for landmark image classification in this comparison. 

### 6.4 Runtime Comparison (on Train set)  
Time per inference step is the average of epoch.  
GPU : Tesla T4  
Epoch : 30  

<p align="center">
 <img width="815" height="193" alt="image" src="https://github.com/user-attachments/assets/01ac8b85-c80a-460a-8ba4-7c712007a680" />
  <br>
</p>
The average training time per epoch varies significantly across models due to differences in their architecture and complexity. InceptionV3 is the fastest (9.41 s) because it has an efficient design with fewer parameters. VGG16 takes the longest time (48.89 s) as it is a large and parameter-heavy model. ResNet50V2 (16.98 s) and NASNetMobile (20.12 s) fall in between, with ResNet50V2 being more optimized. Overall, models with simpler or more efficient architectures train faster on the same GPU. 


### 6.5 Visualizing bubble chart to compare pre-training models with fine-tuning in all aspects
<p align="center">
 <img width="909" height="566" alt="image" src="https://github.com/user-attachments/assets/c0a610e0-d6ef-4f42-9953-bbed20d0a62b" />  
  <br>
</p>
The bubble chart shows the trade-off between training time, model accuracy, and model size. VGG16 has the largest model size, the longest training time per epoch, and relatively lower accuracy compared to the other models. NASNetMobile is smaller and faster but also has the lowest accuracy overall. ResNet50V2 and InceptionV3 achieve the highest accuracy while maintaining much shorter training times, with InceptionV3 being the most efficient. Overall, the chart highlights that larger models like VGG16 are not necessarily more accurate, and more efficient architectures can provide better performance with less computational cost. 

### 6.6 Gread-CAM 
<p align="center">
 <img width="999" height="982" alt="image" src="https://github.com/user-attachments/assets/7d12acfe-6e05-436f-a11b-18f2eeb18d2d" /> 
  <br>
</p>
The Grad-CAM visualization of NASNetMobile shows that the model generally focuses on the key architectural features of each landmark when making correct predictions, such as the main structures of Petra or Christ the Redeemer. However, in misclassified cases, like Chichén Itzá → Great Wall of China and Rome Colosseum → Petra, the attention is less precise and often falls on less distinctive areas, leading to lower confidence. Overall, the results indicate that the model relies on meaningful visual cues but can struggle when the landmark features are ambiguous or partially visible. 

## 7. Discussion and Conclusion


## 8. References
- https://www.researchgate.net/figure/Block-diagram-of-Inception-v3-improved-deep-architecture_fig3_341563435
- https://keras.io/
- https://stackoverflow.com/questions
- https://www.python.org/
- https://pandas.pydata.org/
- https://numpy.org/

## Member, Contribution and Responsibility
### 👥 Team Contributions

| No. | ID | Name | % Contribution | Responsibility |
|:---:|:---:|:----------------------|:---------------:|:-------------------------------------------------------------|
| 1 | 6710422004 | **Chenphop Chanphum ** | 25% | - Collecting data (xx and xx) <br> - Fine-tune Model **VGG16** |
| 2 | 6710422014 | **Nattanon Jiwhanang ** | 25% | - Collecting data (xx and xx) <br> - Fine-tune Model **NASNetMobile** |
| 3 | 6710422029 | **Tanapong Amkwanyeun ** | 25% | - Collecting data (Colosseum and Taj Mahal) <br> - Fine-tune Model **ResNet50V2** |
| 4 | 6710422032 | **Tharathip Khumlert** | 25% | - Collecting data (xx and xx) <br> - Fine-tune Model **xx** |

## End credit  

This project is a part of DADS7202 Deep Learning 

Term: 1 Year of education: 2025 

Master of Science Program in Data Analytics and Data Science (DADS) 

National Institute of Development Administration (NIDA) 

<p align="right">
  <a href="#top">⬆️ Back to top</a>
</p>



