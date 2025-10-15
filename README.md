# Project Name: 7 Wonder Places Classification with CNN

# 1. Introduction
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
---
The main objective is to compare the performance of these four models with and without transfer learning (fine-tuning) to analyze how pre-trained knowledge on large-scale datasets like ImageNet can enhance classification accuracy when adapted to a custom image dataset that the models have never seen before. 
---
Finally, the Grad-CAM (Gradient-weighted Class Activation Mapping) technique is applied to interpret the model’s decision-making process and visualize which image regions most influence the classification. This provides better understanding of model behavior and helps identify potential areas for improvement. 
---
Overall, this study aims to explore the effectiveness of transfer learning and CNN visualization techniques in real-world image classification tasks, contributing to both academic learning and practical implementation in computer vision. 
