# 📰 Fake News Detection System

A comprehensive machine learning system for detecting fake news using multiple classification algorithms with optimized hyperparameters. Built with Gradio for an intuitive web interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-red.svg)

## 🌟 Features

- *4 Classification Models*: Naive Bayes, Passive Aggressive Classifier, SVM, and Logistic Regression
- *2 Preprocessing Methods*: Lemmatization and Stemming with stopword removal
- *Hyperparameter Optimization*: GridSearchCV for optimal model performance
- *Interactive UI*: User-friendly Gradio interface for training and predictions
- *Model Persistence*: Export/import trained models for reusability
- *Comprehensive Metrics*: Accuracy, Precision, Recall, F1-Score with visualizations
- *Real-time Predictions*: Instant fake news detection with confidence scores
- *Pre-trained Models Support*: Deploy with pre-loaded models for immediate use

## 📊 Model Performance

The system compares 8 different configurations (4 models × 2 preprocessing techniques):

| Model | Preprocessing | Typical Accuracy | F1-Score |
|-------|--------------|------------------|----------|
| Logistic Regression | Lemmatization | ~99% | ~0.99 |
| SVM | Stemming | ~99% | ~0.99 |
| Passive Aggressive | Lemmatization | ~98% | ~0.98 |
| Naive Bayes | Stemming | ~94% | ~0.94 |

Note: Performance varies based on dataset and hyperparameters

## 🚀 Quick Start

### Prerequisites

bash
Python 3.8 or higher
pip (Python package manager)


### Installation

1. *Clone the repository*
   bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   

2. *Install dependencies*
   bash
   pip install -r requirements.txt
   

3. *Download the dataset*
   
   Download the ISOT Fake News Dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   
   Extract Fake.csv and True.csv to your working directory

4. *Run the application*
   bash
   python app.py
   

5. *Access the interface*
   
   Open your browser and go to: http://localhost:7860

## 📖 Usage Guide

### Training Models

1. Navigate to the *"🎯 Model Training"* tab
2. Upload Fake.csv and True.csv files
3. Click *"🚀 Train All Models"* (takes 5-10 minutes)
4. View training results and performance comparisons
5. (Optional) Click *"📥 Export All Models"* to save trained models

### Making Predictions

1. Go to the *"🔍 Predict News"* tab
2. Enter or paste news article text
3. Select your preferred model and preprocessing method
4. Click *"🔎 Predict"*
5. View results with confidence scores

### Using Pre-trained Models

*Option 1: Auto-load on startup*
- Place fake_news_models.zip in the same directory as app.py
- Models load automatically when app starts

*Option 2: Manual import*
- Go to "Predict News" tab
- Expand "📂 Import Pre-trained Models"
- Upload your fake_news_models.zip file
- Click "📤 Import Models"

## 🏗 Project Structure


fake-news-detector/
│
├── app.py                      # Main Gradio application
├── fake_news_models.zip        # Pre-trained models (optional)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Exported models directory
│   ├── Naive Bayes_lemma.pkl
│   ├── Naive Bayes_stem.pkl
│   └── ...
│
└── vectorizers/               # TF-IDF vectorizers
    ├── lemma.pkl
    └── stem.pkl


## 🔧 Technical Details

### Data Preprocessing

1. *Text Cleaning*
   - Convert to lowercase
   - Remove special characters and digits
   - Remove extra whitespace

2. *Lemmatization*
   - Reduces words to base/dictionary form
   - Preserves semantic meaning
   - Example: "running" → "run"

3. *Stemming*
   - Reduces words to root form
   - Faster but less accurate
   - Example: "running" → "run"

4. *Feature Extraction*
   - TF-IDF Vectorization
   - Max 5000 features
   - Bigrams (1-2 word combinations)

### Models & Hyperparameters

*Naive Bayes*
- Algorithm: MultinomialNB
- Hyperparameters: alpha = [0.1, 0.5, 1.0]

*Passive Aggressive Classifier*
- Hyperparameters: C = [0.1, 0.5, 1.0], loss = ['hinge', 'squared_hinge']

*Support Vector Machine (SVM)*
- Algorithm: LinearSVC
- Hyperparameters: C = [0.1, 1.0, 10.0], loss = ['hinge', 'squared_hinge']

*Logistic Regression*
- Hyperparameters: C = [0.1, 1.0, 10.0], penalty = ['l2'], solver = ['lbfgs', 'liblinear']

### Evaluation Metrics

- *Accuracy*: Overall correct predictions
- *Precision*: Correct positive predictions / Total positive predictions
- *Recall*: Correct positive predictions / Total actual positives
- *F1-Score*: Harmonic mean of precision and recall

## 🌐 Deployment

### Hugging Face Spaces (Recommended)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose "Gradio" as SDK
3. Upload files: app.py, fake_news_models.zip, requirements.txt
4. Your app will be live automatically!

### Docker

bash
docker build -t fake-news-detector .
docker run -p 7860:7860 fake-news-detector


### Cloud Platforms

Deploy on AWS, GCP, or Azure using the provided deployment guide.

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 📦 Dependencies


gradio>=4.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
nltk>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
tabulate>=0.9.0


## 🎯 Dataset

*ISOT Fake News Dataset*
- Source: [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Contains: ~44,000 articles
- Real News: ~21,000 articles from Reuters.com
- Fake News: ~23,000 articles from unreliable sources
- Features: title, text, subject, date

## 📈 Performance Visualizations

The system generates:
- Accuracy comparison bar charts
- F1-Score comparison charts
- Training time analysis
- Metrics heatmap
- Confidence visualization for predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 🐛 Known Issues & Limitations

- Training takes 5-10 minutes depending on hardware
- Model export file can be 50-200MB in size
- Requires significant RAM (4GB+) for training
- Performance depends on dataset quality and size

## 🔮 Future Enhancements

- [ ] Add deep learning models (BERT, RoBERTa)
- [ ] Implement ensemble methods
- [ ] Add real-time news scraping
- [ ] Multi-language support
- [ ] API endpoint for integration
- [ ] Batch prediction capability
- [ ] Model interpretability (LIME/SHAP)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- ISOT Research Lab for the dataset
- Gradio team for the amazing framework
- scikit-learn community for ML tools
- All contributors and users

## 📧 Contact

For questions or feedback:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

*Made with ❤ for fighting misinformation*

---

## 🔗 Links

- [Live Demo](https://huggingface.co/spaces/yourusername/fake-news-detector)
- [Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [Documentation](https://github.com/yourusername/fake-news-detector/wiki)
- [Report Bug](https://github.com/yourusername/fake-news-detector/issues)
- [Request Feature](https://github.com/yourusername/fake-news-detector/issues)

---

### 📊 Quick Stats


Total Lines of Code: ~600
Number of Models: 8 configurations
Training Time: 5-10 minutes
Prediction Time: <1 second
Accuracy: Up to 99%
