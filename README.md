# Project Name: 7 Wonder Places Classification with CNN

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

Great Wall of China — China 
<img width="310" height="163" alt="image" src="https://github.com/user-attachments/assets/50cab860-9c25-488a-9669-d8496bd85a20" />

Petra — Jordan 
<img width="437" height="203" alt="image" src="https://github.com/user-attachments/assets/e7b3da22-9cc7-4211-ad75-b156c42f64bb" />

Christ the Redeemer — Brazil 
<img width="404" height="269" alt="image" src="https://github.com/user-attachments/assets/19e06ebe-a6a1-49ce-8419-a8c72556e9a9" />

Machu Picchu — Peru 
<img width="463" height="260" alt="image" src="https://github.com/user-attachments/assets/2d59ed7e-7616-44a1-ab65-99fa203e07df" />

Chichen Itza — Mexico 
<img width="470" height="261" alt="image" src="https://github.com/user-attachments/assets/ccd14e66-6334-44c4-8eda-c5a3851bf8e9" />

Colosseum — Italy 
<img width="443" height="303" alt="image" src="https://github.com/user-attachments/assets/69e994b6-ca0c-4d86-a785-ac85cfa6e3e0" />

Taj Mahal - India
<img width="467" height="234" alt="image" src="https://github.com/user-attachments/assets/cbbe599c-3383-4eb1-bb8b-51c2af7aefc3" />

