# 🚦 Traffic Sign Recognition AI: Deep Learning Classifier for GTSRB Dataset

![python-shield](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![tensorflow-shield](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![opencv-shield](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![sklearn-shield](https://img.shields.io/badge/scikit--learn-1.2%2B-blue)
![streamlit-shield](https://img.shields.io/badge/Streamlit-1.25%2B-red)

An **end-to-end deep learning project** that builds, evaluates, and compares multiple CNN architectures to classify German traffic signs from raw images. This repository contains the complete workflow—from comprehensive exploratory data analysis (EDA) and model implementation in Jupyter Notebook to a polished, interactive web application built with Streamlit.

> 💡 The project's key insight is a direct comparison between **Custom CNN Architecture** and **Transfer Learning with VGG19**, revealing optimal approaches for the GTSRB dataset and traffic sign recognition challenges.

---

## 🌟 Key Features

- ✨ **Multi-Model Prediction**: Upload a traffic sign image and get simultaneous predictions from both Custom CNN and VGG19 models with confidence scores.
- 🖼️ **Real-Time Classification**: Test the AI with any traffic sign image (`.png`, `.jpg`, `.jpeg`) and see instant results with visual confidence indicators.
- 📊 **Comprehensive EDA Gallery**: Interactive visualization dashboard showcasing 9+ detailed analyses of the GTSRB dataset, from class distributions to augmentation previews.
- 🏆 **Performance Dashboard**: Compare model architectures, accuracy metrics, and confusion matrices to understand each model's strengths and limitations.
- 🎨 **Modern Web Interface**: A sleek, responsive dark-themed UI with animated elements, traffic light visualization, and professional layout.
- 🚦 **43-Class Classification**: Complete German Traffic Sign Recognition Benchmark (GTSRB) coverage including speed limits, warnings, and regulatory signs.

---

## 🛠️ Tech Stack & Libraries

| Category                | Tools & Libraries                                        |
|-------------------------|----------------------------------------------------------|
| **Data Processing**     | Pandas, NumPy, OpenCV (Image Processing)                |
| **Visualization**       | Matplotlib, Seaborn                                     |
| **Machine Learning**    | Scikit-learn (Preprocessing, Metrics, t-SNE)           |
| **Deep Learning**       | TensorFlow, Keras, VGG19 (Transfer Learning)           |
| **Computer Vision**     | OpenCV, PIL (Python Imaging Library)                    |
| **Web Application**     | Streamlit                                               |
| **Development**         | Jupyter Notebook, Python 3.10+                        |

---

## 📁 Project Structure

```bash
.
├── input/
│   └── Train/
│       ├── 0/           # Speed Limit 20 km/h
│       ├── 1/           # Speed Limit 30 km/h
│       ├── ...          # Classes 0-42
│       └── 42/          # End of no passing by vehicles
├── saved_models/
│   ├── cnn_gtsrb.h5              # Custom CNN trained model
│   └── vgg19_gtsrb.h5            # VGG19 transfer learning model
├── eda_plots/
│   ├── class_distribution.png     # Dataset balance visualization
│   ├── confusion_matrix_cnn.png   # CNN performance matrix
│   ├── model_comparison.png       # Architecture comparison
│   └── (6+ additional EDA plots)
├── traffic_sign_analysis.ipynb    # Main Jupyter Notebook
├── streamlit_app.py               # Streamlit web application
├── style.css                      # Custom CSS styling
└── README.md
```

## ⚙️ Installation & Setup

**1. Clone the Repository** 
```bash
git clone https://github.com/YourUsername/Traffic_Sign_Recognition.git
cd Traffic_Sign_Recognition
```

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/Mac
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install tensorflow opencv-python pillow numpy pandas matplotlib seaborn scikit-learn streamlit
```

**4. Download the GTSRB Dataset**
The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB). Download it from:
- **Kaggle**: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- **Official**: https://benchmark.ini.rub.de/gtsrb_news.html

Extract the dataset and organize it as:
```
input/
└── Train/
    ├── 0/
    ├── 1/
    ...
    └── 42/
```

## ▶️ How to Run the Project

⚠️ **Important:** You must first run the Jupyter notebook to train models and generate visualizations before launching the web application.

**1. Run the Jupyter Notebook** 
```bash
jupyter lab
# or
jupyter notebook
```

Open `traffic_sign_analysis.ipynb` and run all cells sequentially. This will:
- Perform comprehensive EDA with 12+ visualizations
- Train the Custom CNN model (30 epochs)
- Train the VGG19 transfer learning model (15 epochs)
- Save trained models to `saved_models/`
- Generate and save EDA plots to `eda_plots/`

**2. Launch the Streamlit Web Application** 
```bash
streamlit run streamlit_app.py
```

Your browser will automatically open at `http://localhost:8501` with the interactive dashboard.

## 🧠 Model Architectures & Results

Two primary deep learning models were developed and compared for traffic sign classification.

| Model                      | Architecture Type        | Test Accuracy | Training Epochs | Key Characteristics                                              |
|----------------------------|--------------------------|---------------|-----------------|------------------------------------------------------------------|
| **Custom CNN**             | Built from Scratch       | **99.70%**    | 30              | **Best performer**. Optimized for traffic signs with dual conv blocks |
| **VGG19 Transfer Learning** | Pre-trained + Fine-tuned | **59.23%**    | 15              | Struggled with domain transfer from natural images to traffic signs     |

### 🏗️ Custom CNN Architecture
- **Block 1**: 2x Conv2D (32 filters, 5×5) + MaxPooling + Dropout(0.25)
- **Block 2**: 2x Conv2D (64 filters, 3×3) + MaxPooling + Dropout(0.25)
- **Classifier**: Dense(256) + Dropout(0.5) + Dense(43, softmax)
- **Input Size**: 30×30×3 RGB images
- **Optimizer**: Adam (lr=0.001)

### 🎯 VGG19 Transfer Learning
- **Base**: VGG19 pre-trained on ImageNet (frozen)
- **Head**: GlobalAveragePooling2D + Dense(256) + Dropout(0.5) + Dense(43)
- **Input Size**: 48×48×3 RGB images
- **Optimizer**: Adam (lr=0.0001)

## 📊 Dataset Statistics & EDA Highlights

- **Total Images**: 39,209 training samples
- **Classes**: 43 German traffic sign categories
- **Image Sizes**: Variable (resized to 30×30 for CNN, 48×48 for VGG19)
- **Data Augmentation**: Rotation (±10°), zoom (±10%), shifting, shearing
- **Class Balance**: Mild imbalance handled through stratified splitting

### Key EDA Insights:
- **Class Distribution**: Some classes have 10x more samples than others
- **Image Quality**: High variance in brightness, blur, and aspect ratios
- **Augmentation Impact**: Significant improvement in model generalization
- **Feature Separability**: t-SNE reveals moderate class clustering in pixel space

## 🎯 Performance Metrics

| Metric                | Custom CNN | VGG19      |
|-----------------------|-----------|------------|
| **Test Accuracy**     | 99.70%    | 59.23%     |
| **Training Time**     | ~45 mins  | ~25 mins   |
| **Model Size**        | 2.3 MB    | 78 MB      |
| **Inference Speed**   | Fast      | Moderate   |
| **Memory Usage**      | Low       | High       |

✅ The **Custom CNN** significantly outperforms VGG19 transfer learning, demonstrating that domain-specific architectures often excel over general-purpose pre-trained models for specialized tasks like traffic sign recognition.

## 🌐 Web Application Features

The Streamlit dashboard provides:
- **🔮 AI Prediction**: Real-time traffic sign classification
- **📊 EDA Gallery**: Interactive data visualization explorer
- **🧠 Model Comparison**: Side-by-side architecture and performance analysis
- **🎯 Confidence Scores**: Visual confidence indicators for predictions
- **🚦 Animated UI**: Traffic light themed interface with smooth transitions

## 🤝 Contributing

We welcome contributions to improve the Traffic Sign Recognition system!

**1.** Fork the repository

**2.** Create your feature branch
```bash
git checkout -b feature/AmazingFeature
```

**3.** Commit your changes
```bash
git commit -m "Add some AmazingFeature"
```

**4.** Push to your branch
```bash
git push origin feature/AmazingFeature
```

**5.** Open a Pull Request

### Areas for Contribution:
- Model optimization and new architectures
- Real-time video processing capabilities
- Mobile app development
- Dataset expansion with other countries' traffic signs
- Performance optimization and model compression

## 🔮 Future Enhancements

- [ ] **Real-time Video Processing**: Webcam integration for live traffic sign detection
- [ ] **Mobile App**: Flutter/React Native app for on-the-go recognition
- [ ] **Multi-Country Support**: Expand to US, UK, and other traffic sign systems
- [ ] **Object Detection**: Combine with YOLO for sign detection + classification
- [ ] **Model Compression**: TensorFlow Lite optimization for edge deployment
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy

## 📚 References & Dataset

- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Paper**: Stallkamp et al. "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition"
- **Competition**: Originally from IJCNN 2011 competition
- **Classes**: Based on German StVO (Road Traffic Regulations)

## 📧 Contact

**Your Name**

📫 **your.email@example.com**
🔗 **GitHub**: [@YourUsername](https://github.com/YourUsername)
🌐 **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

⭐ **Star this repository if you found it helpful!** ⭐
