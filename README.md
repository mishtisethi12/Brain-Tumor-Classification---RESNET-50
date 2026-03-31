# Brain-Tumor-Classification---RESNET-50
A Computer Vision project using ResNet-50 and deep learning to classify brain MRI images into tumor categories for automated medical diagnosis.

Abstract
This project presents the design and implementation of a Brain Tumor Classification system using deep learning techniques. The system utilizes a pre-trained ResNet-50 model to classify MRI brain images into different tumor categories such as glioma, meningioma, pituitary tumor, and no tumor. The aim is to assist in automated medical diagnosis by providing accurate and efficient predictions.

 Problem Statement
Brain tumor detection using MRI images is a challenging and critical task in the medical field. Manual analysis is time-consuming and prone to errors. This project aims to develop an automated system that can accurately classify MRI images into tumor categories using deep learning techniques.

 Objectives
- To implement a deep learning model using ResNet-50  
- To preprocess MRI images for better performance  
- To classify brain tumors into multiple categories  
- To reduce misclassification errors  
- To enable command-line execution  

Dataset
The dataset consists of MRI brain images categorized into:
- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

(Dataset can be sourced from publicly available datasets such as Kaggle Brain MRI datasets.)

Project Structure
brain-tumor-classification/
│── data/
│── models/
│   ├── brain_tumor_resnet50.pth
│   ├── brain_tumor_resnet50_final.pth
│── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│── src/
│   ├── train.py
│   ├── test.py
│   ├── predict.py
│   ├── results.py
│── requirements.txt
│── README.md
│── report.pdf

Installation
Step 1: Clone the repository
git clone https://github.com/YOUR-USERNAME/brain-tumor-classification.git
cd brain-tumor-classification

Step 2: Install dependencies
pip install -r requirements.txt

How to Run

Train the Model
python src/train.py

Test the Model
python src/test.py

Predict on a Single Image
python src/predict.py --image path_to_image.jpg

 Results
The model was evaluated on MRI images and achieved good classification accuracy.

Outputs include:
- Confusion Matrix  
- ROC Curve  

These results are available in the outputs/ folder.

Observations
- The model performs well on clearly distinguishable MRI images  
- Initial bias toward a single class was reduced after tuning  
- Performance improves with better data balancing  

 Advantages
- Automated tumor classification  
- Reduces manual effort  
- Fast and efficient predictions  
- Scalable system  

Limitations
- Requires large dataset  
- Sensitive to class imbalance  
- May misclassify similar tumor types  
- Requires computational resources  

 Future Scope
- Improve accuracy using advanced models  
- Deploy as web or mobile application  
- Integrate with healthcare systems  
- Add tumor segmentation  

Conclusion
This project demonstrates the application of deep learning in medical image classification. By leveraging ResNet-50 and transfer learning, the system can effectively classify brain MRI images. The project provides a strong foundation for further improvements and real-world deployment.

