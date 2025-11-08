import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import gradio as gr
import pickle
import os
import zipfile
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Global variables for models and vectorizers
trained_models = {}
vectorizers = {}
training_results = None
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Text preprocessing functions
def clean_text(text):
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_with_lemmatization(text):
    """Preprocess text with lemmatization"""
    text = clean_text(text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_with_stemming(text):
    """Preprocess text with stemming"""
    text = clean_text(text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_pretrained_models():
    """Load pre-trained models from zip file if available"""
    global trained_models, vectorizers, training_results
    
    model_zip_path = 'fake_news_models.zip'
    
    if os.path.exists(model_zip_path):
        try:
            print("Loading pre-trained models...")
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                # Load models
                model_files = [f for f in zip_ref.namelist() if f.startswith('models/')]
                for model_file in model_files:
                    model_name = os.path.basename(model_file).replace('.pkl', '')
                    model_data = zip_ref.read(model_file)
                    trained_models[model_name] = pickle.loads(model_data)
                
                # Load vectorizers
                vec_files = [f for f in zip_ref.namelist() if f.startswith('vectorizers/')]
                for vec_file in vec_files:
                    vec_name = os.path.basename(vec_file).replace('.pkl', '')
                    vec_data = zip_ref.read(vec_file)
                    vectorizers[vec_name] = pickle.loads(vec_data)
                
                # Load training results if available
                if 'training_results.csv' in zip_ref.namelist():
                    results_data = zip_ref.read('training_results.csv')
                    training_results = pd.read_csv(BytesIO(results_data))
            
            print(f"✅ Successfully loaded {len(trained_models)} pre-trained models!")
            return True
        except Exception as e:
            print(f"❌ Error loading pre-trained models: {str(e)}")
            return False
    else:
        print("ℹ️ No pre-trained models found. You'll need to train or import models.")
        return False

def train_models(fake_file, real_file, progress=gr.Progress()):
    """Train all models with hyperparameter optimization"""
    global trained_models, vectorizers, training_results
    
    try:
        progress(0, desc="Loading datasets...")
        # Load datasets
        fake_df = pd.read_csv(fake_file.name)
        real_df = pd.read_csv(real_file.name)
        
        # Add labels
        fake_df['label'] = 0  # Fake news
        real_df['label'] = 1  # Real news
        
        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        df['content'] = df['title'] + ' ' + df['text']
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        progress(0.1, desc="Preprocessing with lemmatization...")
        df['content_lemma'] = df['content'].apply(preprocess_with_lemmatization)
        
        progress(0.2, desc="Preprocessing with stemming...")
        df['content_stem'] = df['content'].apply(preprocess_with_stemming)
        
        # Prepare data
        X_lemma = df['content_lemma']
        X_stem = df['content_stem']
        y = df['label']
        
        # Split data
        X_train_lemma, X_test_lemma, y_train, y_test = train_test_split(
            X_lemma, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_stem, X_test_stem, _, _ = train_test_split(
            X_stem, y, test_size=0.2, random_state=42, stratify=y
        )
        
        progress(0.3, desc="Vectorizing text...")
        # TF-IDF Vectorization
        tfidf_lemma = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf_lemma = tfidf_lemma.fit_transform(X_train_lemma)
        X_test_tfidf_lemma = tfidf_lemma.transform(X_test_lemma)
        
        tfidf_stem = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf_stem = tfidf_stem.fit_transform(X_train_stem)
        X_test_tfidf_stem = tfidf_stem.transform(X_test_stem)
        
        vectorizers['lemma'] = tfidf_lemma
        vectorizers['stem'] = tfidf_stem
        
        # Define models
        models = {
            'Naive Bayes': {
                'model': MultinomialNB(),
                'params': {'alpha': [0.1, 0.5, 1.0]}
            },
            'Passive Aggressive': {
                'model': PassiveAggressiveClassifier(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 0.5, 1.0], 'loss': ['hinge', 'squared_hinge']}
            },
            'SVM': {
                'model': LinearSVC(random_state=42, max_iter=2000),
                'params': {'C': [0.1, 1.0, 10.0], 'loss': ['hinge', 'squared_hinge']}
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']}
            }
        }
        
        results = []
        total_models = len(models) * 2
        current_model = 0
        
        # Train with lemmatization
        for model_name, model_config in models.items():
            current_model += 1
            progress(0.3 + (current_model / total_models) * 0.6, 
                    desc=f"Training {model_name} (Lemmatization)...")
            
            grid_search = GridSearchCV(
                model_config['model'], model_config['params'], 
                cv=3, scoring='f1', n_jobs=-1, verbose=0
            )
            
            start_time = time()
            grid_search.fit(X_train_tfidf_lemma, y_train)
            training_time = time() - start_time
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_tfidf_lemma)
            
            trained_models[f"{model_name}_lemma"] = best_model
            
            results.append({
                'Model': model_name,
                'Preprocessing': 'Lemmatization',
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'Training Time (s)': round(training_time, 2),
                'Best Params': str(grid_search.best_params_)
            })
        
        # Train with stemming
        for model_name, model_config in models.items():
            current_model += 1
            progress(0.3 + (current_model / total_models) * 0.6,
                    desc=f"Training {model_name} (Stemming)...")
            
            grid_search = GridSearchCV(
                model_config['model'], model_config['params'],
                cv=3, scoring='f1', n_jobs=-1, verbose=0
            )
            
            start_time = time()
            grid_search.fit(X_train_tfidf_stem, y_train)
            training_time = time() - start_time
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_tfidf_stem)
            
            trained_models[f"{model_name}_stem"] = best_model
            
            results.append({
                'Model': model_name,
                'Preprocessing': 'Stemming',
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'Training Time (s)': round(training_time, 2),
                'Best Params': str(grid_search.best_params_)
            })
        
        progress(0.95, desc="Creating visualizations...")
        training_results = pd.DataFrame(results)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        pivot_acc = training_results.pivot(index='Model', columns='Preprocessing', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].legend(title='Preprocessing')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # F1-Score comparison
        pivot_f1 = training_results.pivot(index='Model', columns='Preprocessing', values='F1-Score')
        pivot_f1.plot(kind='bar', ax=axes[0, 1], color=['#3498db', '#f39c12'])
        axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].legend(title='Preprocessing')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Training time
        pivot_time = training_results.pivot(index='Model', columns='Preprocessing', values='Training Time (s)')
        pivot_time.plot(kind='bar', ax=axes[1, 0], color=['#9b59b6', '#1abc9c'])
        axes[1, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].legend(title='Preprocessing')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Metrics heatmap
        metrics_pivot = training_results.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean()
        sns.heatmap(metrics_pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[1, 1])
        axes[1, 1].set_title('Average Metrics Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        progress(1.0, desc="Training complete!")
        
        return (
            training_results.to_html(index=False),
            fig,
            "✅ Training completed successfully! You can now use the prediction tab."
        )
        
    except Exception as e:
        return (
            f"❌ Error during training: {str(e)}",
            None,
            f"❌ Training failed: {str(e)}"
        )

def predict_news(text, model_choice, preprocessing_choice):
    """Predict if news is fake or real"""
    global trained_models, vectorizers
    
    if not trained_models:
        return "⚠️ Please train the models first using the Training tab or load pre-trained models!", None
    
    try:
        # Preprocess text
        if preprocessing_choice == "Lemmatization":
            processed_text = preprocess_with_lemmatization(text)
            vectorizer = vectorizers['lemma']
            key_suffix = '_lemma'
        else:
            processed_text = preprocess_with_stemming(text)
            vectorizer = vectorizers['stem']
            key_suffix = '_stem'
        
        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])
        
        # Get model
        model_key = f"{model_choice}{key_suffix}"
        model = trained_models[model_key]
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vectorized)[0]
            confidence = max(proba) * 100
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(text_vectorized)[0]
            confidence = (1 / (1 + np.exp(-decision))) * 100 if prediction == 1 else (1 / (1 + np.exp(decision))) * 100
        else:
            confidence = None
        
        result = "✅ REAL NEWS" if prediction == 1 else "❌ FAKE NEWS"
        
        if confidence:
            result += f"\n\nConfidence: {confidence:.2f}%"
        
        result += f"\n\nModel: {model_choice}\nPreprocessing: {preprocessing_choice}"
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        categories = ['Fake News', 'Real News']
        colors = ['#e74c3c', '#2ecc71']
        
        if confidence:
            if prediction == 1:
                values = [100 - confidence, confidence]
            else:
                values = [confidence, 100 - confidence]
            
            bars = ax.barh(categories, values, color=colors, alpha=0.7)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Confidence (%)', fontsize=12)
            ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
            
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Prediction: {result.split()[1]}', 
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   transform=ax.transAxes)
            ax.axis('off')
        
        plt.tight_layout()
        
        return result, fig
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", None

def get_model_details():
    """Return detailed information about trained models"""
    if training_results is None:
        return "⚠️ No training results available. Please train the models first."
    
    best_model = training_results.loc[training_results['F1-Score'].idxmax()]
    
    details = f"""
## 🏆 Best Performing Model

**Model:** {best_model['Model']}  
**Preprocessing:** {best_model['Preprocessing']}  
**Accuracy:** {best_model['Accuracy']:.4f}  
**Precision:** {best_model['Precision']:.4f}  
**Recall:** {best_model['Recall']:.4f}  
**F1-Score:** {best_model['F1-Score']:.4f}  
**Training Time:** {best_model['Training Time (s)']} seconds  
**Best Parameters:** {best_model['Best Params']}

---

## 📊 All Models Performance

{training_results.to_markdown(index=False)}
"""
    return details

def export_models():
    """Export all trained models and vectorizers to a zip file"""
    if not trained_models:
        return None, "⚠️ No trained models to export. Please train the models first."
    
    try:
        # Create a BytesIO object to store the zip file
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save all trained models
            for model_name, model in trained_models.items():
                model_bytes = pickle.dumps(model)
                zip_file.writestr(f'models/{model_name}.pkl', model_bytes)
            
            # Save vectorizers
            for vec_name, vectorizer in vectorizers.items():
                vec_bytes = pickle.dumps(vectorizer)
                zip_file.writestr(f'vectorizers/{vec_name}.pkl', vec_bytes)
            
            # Save training results
            if training_results is not None:
                results_csv = training_results.to_csv(index=False)
                zip_file.writestr('training_results.csv', results_csv)
        
        zip_buffer.seek(0)
        
        # Save to temporary file
        temp_path = 'fake_news_models.zip'
        with open(temp_path, 'wb') as f:
            f.write(zip_buffer.getvalue())
        
        return temp_path, "✅ Models exported successfully!"
    
    except Exception as e:
        return None, f"❌ Error exporting models: {str(e)}"

def import_models(zip_file):
    """Import trained models and vectorizers from a zip file"""
    global trained_models, vectorizers, training_results
    
    if zip_file is None:
        return "⚠️ Please upload a models zip file.", gr.update(interactive=False)
    
    try:
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            # Load models
            model_files = [f for f in zip_ref.namelist() if f.startswith('models/')]
            for model_file in model_files:
                model_name = os.path.basename(model_file).replace('.pkl', '')
                model_data = zip_ref.read(model_file)
                trained_models[model_name] = pickle.loads(model_data)
            
            # Load vectorizers
            vec_files = [f for f in zip_ref.namelist() if f.startswith('vectorizers/')]
            for vec_file in vec_files:
                vec_name = os.path.basename(vec_file).replace('.pkl', '')
                vec_data = zip_ref.read(vec_file)
                vectorizers[vec_name] = pickle.loads(vec_data)
            
            # Load training results if available
            if 'training_results.csv' in zip_ref.namelist():
                results_data = zip_ref.read('training_results.csv')
                training_results = pd.read_csv(BytesIO(results_data))
        
        return f"✅ Successfully imported {len(trained_models)} models and {len(vectorizers)} vectorizers!", gr.update(interactive=True)
    
    except Exception as e:
        return f"❌ Error importing models: {str(e)}", gr.update(interactive=False)

def get_initial_status():
    """Get initial status message based on whether models are loaded"""
    if trained_models:
        return f"✅ {len(trained_models)} pre-trained models loaded and ready!", True
    else:
        return "ℹ️ No pre-trained models found. Please train models or import them.", False

# Load pre-trained models at startup
models_loaded = load_pretrained_models()

# Create Gradio interface
with gr.Blocks(title="Fake News Detection System", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 📰 Fake News Detection System
    
    This system compares 4 machine learning models (Naive Bayes, Passive Aggressive, SVM, Logistic Regression) 
    with 2 preprocessing techniques (Lemmatization and Stemming) for fake news classification.
    """)
    
    # Status indicator
    initial_status, predict_enabled = get_initial_status()
    status_box = gr.Markdown(f"### {initial_status}")
    
    with gr.Tabs() as tabs:
        # Training Tab
        with gr.Tab("🎯 Model Training"):
            gr.Markdown("### Upload ISOT Dataset Files")
            gr.Markdown("Download the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")
            
            with gr.Row():
                fake_file = gr.File(label="Upload Fake.csv", file_types=[".csv"])
                real_file = gr.File(label="Upload True.csv", file_types=[".csv"])
            
            train_btn = gr.Button("🚀 Train All Models", variant="primary", size="lg")
            
            training_status = gr.Markdown("")
            
            with gr.Row():
                results_table = gr.HTML(label="Training Results")
            
            with gr.Row():
                results_plot = gr.Plot(label="Performance Comparison")
            
            gr.Markdown("---")
            gr.Markdown("### 💾 Export Trained Models")
            export_btn = gr.Button("📥 Export All Models", variant="secondary")
            export_status = gr.Markdown("")
            export_file = gr.File(label="Download Models", interactive=False)
            
            export_btn.click(
                fn=export_models,
                outputs=[export_file, export_status]
            )
        
        # Prediction Tab
        with gr.Tab("🔍 Predict News"):
            gr.Markdown("### Enter news text to check if it's fake or real")
            
            with gr.Accordion("📂 Import Pre-trained Models", open=False):
                gr.Markdown("Upload a previously exported models zip file to use pre-trained models")
                import_file = gr.File(label="Upload Models Zip File", file_types=[".zip"])
                import_btn = gr.Button("📤 Import Models", variant="secondary")
                import_status = gr.Markdown("")
            
            gr.Markdown("---")
            
            news_text = gr.Textbox(
                label="News Text",
                placeholder="Enter the news article text here...",
                lines=8
            )
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["Naive Bayes", "Passive Aggressive", "SVM", "Logistic Regression"],
                    label="Select Model",
                    value="Logistic Regression"
                )
                
                preprocessing_selector = gr.Dropdown(
                    choices=["Lemmatization", "Stemming"],
                    label="Select Preprocessing",
                    value="Lemmatization"
                )
            
            predict_btn = gr.Button("🔎 Predict", variant="primary", size="lg", interactive=predict_enabled)
            
            with gr.Row():
                prediction_output = gr.Textbox(label="Prediction Result", lines=5)
            
            with gr.Row():
                prediction_plot = gr.Plot(label="Confidence Visualization")
            
            # Example texts
            gr.Markdown("### 📝 Example News Articles")
            gr.Examples(
                examples=[
                    ["President announces new economic reforms to boost GDP growth and create millions of jobs in the manufacturing sector."],
                    ["BREAKING: Scientists discover cure for all diseases using this one weird trick! Doctors hate this!"],
                    ["The Federal Reserve raised interest rates by 0.25% to combat inflation, according to official statements."]
                ],
                inputs=news_text
            )
            
            predict_btn.click(
                fn=predict_news,
                inputs=[news_text, model_selector, preprocessing_selector],
                outputs=[prediction_output, prediction_plot]
            )
        
        # Model Details Tab
        with gr.Tab("📊 Model Details"):
            gr.Markdown("### Detailed Performance Metrics")
            
            refresh_btn = gr.Button("🔄 Refresh Details", variant="secondary")
            details_output = gr.Markdown()
            
            # Auto-load details if models are available
            if models_loaded and training_results is not None:
                details_output.value = get_model_details()
            
            refresh_btn.click(
                fn=get_model_details,
                outputs=details_output
            )
    
    # Set up training button click event
    train_btn.click(
        fn=train_models,
        inputs=[fake_file, real_file],
        outputs=[results_table, results_plot, training_status]
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=predict_btn
    ).then(
        fn=lambda: "✅ Models trained and ready for predictions!",
        outputs=status_box
    )
    
    # Set up import button click event
    import_btn.click(
        fn=import_models,
        inputs=import_file,
        outputs=[import_status, predict_btn]
    ).then(
        fn=lambda: "✅ Models imported and ready for predictions!",
        outputs=status_box
    )

if __name__ == "__main__":
    app.launch(share=True)