# Sign Language Recognition System

## **Project Overview**
This project is a **real-time sign language recognition system** that leverages **OpenCV, MediaPipe, and TensorFlow** to detect and classify hand gestures used in sign language communication. The model is trained to recognize various sign language gestures using a deep learning approach.

## **Features**
 Real-time hand tracking using **MediaPipe**  
 Sign gesture classification using **TensorFlow**  
 Multi-threaded processing for optimized performance  
 Flask-based web interface for live recognition  
 CPU-optimized with optional GPU acceleration  

## **Installation**
To set up the project, follow these steps:

### **1. Clone the Repository**
```bash
git clone https://github.com/cogniworks12/Sign-Language-Recognition.git
cd Sign-Language-Recognition
```

### **2. Create and Activate a Virtual Environment**
```bash
python3 -m venv env
source env/bin/activate  # For Linux/macOS
env\Scripts\activate  # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Application**
```bash
python app.py
```


## **Usage**
1. Ensure your webcam is connected.
2. Run the script and make sign language gestures in front of the camera.
3. The recognized sign will be displayed on the screen.

## **Model Training**
The model is trained using **TensorFlow and Keras** with a dataset of sign language gestures. The dataset includes multiple variations to improve accuracy.

## **Supported Signs**
This system currently supports a variety of sign language gestures. Additional signs can be added by retraining the model with extended datasets.

## **Contributing**
Pull requests are welcome! If you encounter issues or have suggestions for improvements, feel free to contribute.

## **License**
This project is licensed under the **MIT License**.

---
**Author:** Cogniworks.ai  
🔗 [GitHub](https://github.com/cogniworks12)  
🚀 Powered by AI & Deep Learning

