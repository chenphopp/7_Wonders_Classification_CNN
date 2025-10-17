Project Name: 7 Wonder Places Classification with CNN
 
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

### Data Sources
The images were collected from open-source image repositories such as Unsplash, Google Images using keyword-based web scraping. Each category contains a diverse set of photos that vary in lighting conditions, camera angles, weather, and distance, ensuring that the model can generalize across real-world scenarios.

### Imbalance dataset
Check amount of images per classes  
<p align="center">
  <img width="500" height="292" alt="image" src="https://github.com/user-attachments/assets/aad620ea-f0ca-469b-89fa-f961b6313f29" />
  <br>
  <img width="1790" height="489" alt="image" src="https://github.com/user-attachments/assets/036566db-ab2a-4bfc-a65b-45f846145fb3" />
</p>

Check image quality per classes  
<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/f560781b-eab6-4623-b771-4f6e9e6967f5" alt="Train dataset" width="400"/>
    <img src="https://github.com/user-attachments/assets/e859de4e-a403-431f-938d-1561c24f89a4" alt="Validation dataset" width="400"/>
</div>

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/581e144b-e19c-4e63-b10b-469f9b396277" alt="Test dataset" width="400"/>
</div>

## 3. Data Preparation and Pre-Processing    
Before training, all images were preprocessed as follows:  
- Resizing: Each image was resized to 224×224 pixels to fit the input shape of pre-trained CNN models.
  

- Augmentation technique were applied to increase dataset diversity and reduce overfitting.  

<table align="center">
  <tr>
    <td colspan="2" align="center">
       <img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/a1db14a9-bde7-4037-8c56-23a089ac8a68">
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

## 4.  Model architecture

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

- Network Architecture of Pre-training model without fine-tuning vs with Fine tuning - VGG16

<img width="1342" height="412" alt="image" src="https://github.com/user-attachments/assets/1b7ff7a1-39d2-41a1-b10c-360731c275c0" />

- Network Architecture of Pre-training with and without out Fine-tuning - **InceptionV3**

<img width="304" height="676" alt="image" src="https://github.com/user-attachments/assets/5a90830c-5d8e-4f7a-abf1-4e18f27ef194" />

### 4.4 NASNetMobile

## 5. Training method
*Training Configuration*  
The table below summarizes the key hyperparameter settings used for training all CNN models in both cases — before and after fine-tuning.
All models were trained using the same optimizer and loss function to ensure consistent evaluation conditions.  

<img width="1106" height="211" alt="image" src="https://github.com/user-attachments/assets/6072c730-f129-4724-9332-d911fd4e8afc" />

