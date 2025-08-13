import streamlit as st
import os
from PIL import Image
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

# Page config
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load models
@st.cache_resource
def load_models():
    try:
        cnn_model = load_model("saved_models/cnn_gtsrb.h5")
        vgg_model = load_model("saved_models/vgg19_gtsrb.h5")
        return cnn_model, vgg_model
    except:
        return None, None

def preprocess_image_cnn(image):
    img = cv2.resize(np.array(image), (30, 30))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_image_vgg(image):
    img = cv2.resize(np.array(image), (48, 48))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Traffic sign class names (GTSRB)
SIGN_NAMES = {
    0: 'Speed Limit 20', 1: 'Speed Limit 30', 2: 'Speed Limit 50', 3: 'Speed Limit 60',
    4: 'Speed Limit 70', 5: 'Speed Limit 80', 6: 'End Speed Limit 80', 7: 'Speed Limit 100',
    8: 'Speed Limit 120', 9: 'No Passing', 10: 'No Passing Vehicles', 11: 'Right of Way',
    12: 'Priority Road', 13: 'Yield', 14: 'Stop', 15: 'No Vehicles', 16: 'Vehicles > 3.5t Prohibited',
    17: 'No Entry', 18: 'General Caution', 19: 'Dangerous Curve Left', 20: 'Dangerous Curve Right',
    21: 'Double Curve', 22: 'Bumpy Road', 23: 'Slippery Road', 24: 'Road Narrows Right',
    25: 'Road Work', 26: 'Traffic Signals', 27: 'Pedestrians', 28: 'Children Crossing',
    29: 'Bicycles Crossing', 30: 'Beware Ice/Snow', 31: 'Wild Animals Crossing',
    32: 'End Speed/Passing Limits', 33: 'Turn Right Ahead', 34: 'Turn Left Ahead',
    35: 'Ahead Only', 36: 'Go Straight/Right', 37: 'Go Straight/Left', 38: 'Keep Right',
    39: 'Keep Left', 40: 'Roundabout', 41: 'End No Passing', 42: 'End No Passing Vehicles'
}

# Main app
def main():
    load_css()
    
    # Navigation
    st.markdown("""
    <nav class="navbar">
        <div class="nav-container">
            <div class="nav-logo">🚦 Traffic AI</div>
            <div class="nav-links">
                <a href="#home" class="nav-link">Home</a>
                <a href="#predict" class="nav-link">Predict</a>
                <a href="#analysis" class="nav-link">Analysis</a>
                <a href="#models" class="nav-link">Models</a>
                <a href="#results" class="nav-link">Results</a>
            </div>
        </div>
    </nav>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section" id="home">
        <div class="hero-content">
            <div class="hero-text">
                <h1 class="hero-title">
                    <span class="text-gradient">Traffic Sign</span><br>
                    <span class="text-gradient">Recognition AI</span>
                </h1>
                <p class="hero-subtitle">
                    Deep Learning powered classification system for German Traffic Sign Recognition Benchmark (GTSRB)
                </p>
                <div class="stats-container">
                    <div class="stat-card">
                        <div class="stat-number">43</div>
                        <div class="stat-label">Classes</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">39K+</div>
                        <div class="stat-label">Images</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">99%</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                </div>
            </div>
            <div class="hero-visual">
                <div class="traffic-light-system">
                    <div class="traffic-light-pole"></div>
                    <div class="traffic-light-container">
                        <div class="traffic-light">
                            <div class="light red-light" id="red"></div>
                            <div class="light yellow-light" id="yellow"></div>
                            <div class="light green-light" id="green"></div>
                        </div>
                        <div class="light-glow red-glow"></div>
                        <div class="light-glow yellow-glow"></div>
                        <div class="light-glow green-glow"></div>
                    </div>
                    <div class="scanning-beam"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction Section
    st.markdown('<div class="section-divider" id="predict"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🔮 AI Prediction</h2>', unsafe_allow_html=True)
    
    # Load models
    cnn_model, vgg_model = load_models()
    
    if cnn_model is None or vgg_model is None:
        st.markdown("""
        <div class="prediction-container">
            <div class="prediction-error">
                ⚠️ Models not found! Please ensure models are saved in 'saved_models/' directory.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        
        # File uploader with custom styling
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<div class="upload-title">Upload Traffic Sign Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "", 
            type=['png', 'jpg', 'jpeg'],
            label_visibility="hidden"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                st.image(image, width=200, caption="Uploaded Image")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Predict with both models
            with st.spinner('🔍 Analyzing image...'):
                # CNN Prediction
                cnn_input = preprocess_image_cnn(image)
                cnn_prediction = cnn_model.predict(cnn_input, verbose=0)
                cnn_class = np.argmax(cnn_prediction)
                cnn_confidence = float(np.max(cnn_prediction) * 100)
                
                # VGG Prediction
                vgg_input = preprocess_image_vgg(image)
                vgg_prediction = vgg_model.predict(vgg_input, verbose=0)
                vgg_class = np.argmax(vgg_prediction)
                vgg_confidence = float(np.max(vgg_prediction) * 100)
            
            # Display predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-result cnn-result">
                    <div class="model-name">🔥 Custom CNN</div>
                    <div class="prediction-sign">{SIGN_NAMES.get(cnn_class, 'Unknown')}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill cnn-fill" style="width: {cnn_confidence}%"></div>
                        <div class="confidence-text">{cnn_confidence:.1f}% Confidence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-result vgg-result">
                    <div class="model-name">🎯 VGG19</div>
                    <div class="prediction-sign">{SIGN_NAMES.get(vgg_class, 'Unknown')}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill vgg-fill" style="width: {vgg_confidence}%"></div>
                        <div class="confidence-text">{vgg_confidence:.1f}% Confidence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # EDA Section
    st.markdown('<div class="section-divider" id="analysis"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Plot configuration
    plots_folder = "eda_plots"
    plot_files = [
        ("Class Distribution", "class_distribution(images_per_class).png"),
        ("Augmentation Preview", "augmentation_preview.png"),
        ("CNN Accuracy & Loss", "cnn_accuracy_loss.png"),
        ("Confusion Matrix (CNN)", "confusion_matrix_cnn.png"),
        ("Duplicate Groups", "duplicate_groups.png"),
        ("Median Size", "Median_Size.png"),
        ("RGB Mean", "RGB_Mean.png"),
        ("t-SNE", "t-sne.png"),
        ("Model Comparison", "model_comparison.png")
    ]
    
    # Create plot grid
    for i in range(0, len(plot_files), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(plot_files):
                title, filename = plot_files[i + j]
                filepath = os.path.join(plots_folder, filename)
                
                if os.path.exists(filepath):
                    with col:
                        st.markdown(f"""
                        <div class="plot-card">
                            <div class="plot-title">{title}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        img = Image.open(filepath)
                        # Handle specific plots that need height control
                        if filename in ["confusion_matrix_cnn.png", "t-sne.png"]:
                            st.markdown('<div class="plot-container-small">', unsafe_allow_html=True)
                        st.image(img, use_container_width=True)
                        if filename in ["confusion_matrix_cnn.png", "t-sne.png"]:
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    with col:
                        st.markdown(f"""
                        <div class="plot-card error">
                            <div class="plot-title">{title}</div>
                            <p>Plot not found: {filename}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Model Architecture Section
    st.markdown('<div class="section-divider" id="models"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🧠 Model Architecture</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3 class="model-title">🔥 Custom CNN</h3>
            <div class="model-details">
                <div class="layer">Conv2D (32 filters, 5×5)</div>
                <div class="layer">Conv2D (32 filters, 5×5)</div>
                <div class="layer">MaxPooling2D (2×2)</div>
                <div class="layer">Conv2D (64 filters, 3×3)</div>
                <div class="layer">Conv2D (64 filters, 3×3)</div>
                <div class="layer">MaxPooling2D (2×2)</div>
                <div class="layer">Dense (256 units)</div>
                <div class="layer">Dense (43 classes)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3 class="model-title">🎯 VGG19 Transfer Learning</h3>
            <div class="model-details">
                <div class="layer">VGG19 Base (Frozen)</div>
                <div class="layer">GlobalAveragePooling2D</div>
                <div class="layer">Dense (256 units)</div>
                <div class="layer">Dropout (0.5)</div>
                <div class="layer">Dense (43 classes)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Results Section
    st.markdown('<div class="section-divider" id="results"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🎯 Results & Performance</h2>', unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card green">
            <div class="metric-value">99.7%</div>
            <div class="metric-label">CNN Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card yellow">
            <div class="metric-value">59.23%</div>
            <div class="metric-label">VGG19 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card red">
            <div class="metric-value">0.15</div>
            <div class="metric-label">Best Loss</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card green">
            <div class="metric-value">30</div>
            <div class="metric-label">Epochs</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <p>🚦 Traffic Sign Recognition AI Dashboard</p>
            <p>Powered by Deep Learning & Streamlit</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()