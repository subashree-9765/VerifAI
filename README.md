# VerifAI - Fake News Style Analysis System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Status](https://img.shields.io/badge/Status-Active-success)

## Overview

Fake news detection is a popular Natural Language Processing (NLP) challenge. Many projects claim to determine whether news content is true or false, but text classification models alone cannot verify factual accuracy.

This project approaches the problem differently. Instead of acting as a fact-checking engine, it analyzes linguistic patterns and writing characteristics to estimate whether a piece of text resembles examples labeled as real or fake within a training dataset.

The system is designed as an educational machine learning project that demonstrates responsible AI development, transparent evaluation, and modern NLP workflows.

---

## Key Features

* TF-IDF text vectorization
* Logistic Regression classifier
* Streamlit web application
* Command-line prediction interface
* Confidence scoring system
* REAL / FAKE / UNCERTAIN classifications
* Model evaluation metrics
* Automated chart generation
* Dataset leakage analysis
* Unit testing with Pytest
* Continuous Integration support
* Saved model artifacts for deployment

---

## Project Goals

The primary objectives of this project are:

* Demonstrate a complete machine learning workflow
* Showcase NLP classification techniques
* Provide an interactive prediction dashboard
* Highlight responsible AI practices
* Present reproducible experimentation
* Create a portfolio-ready ML application

---

## What the System Can Do

The application can:

* Analyze news headlines
* Analyze article excerpts
* Generate prediction probabilities
* Classify text into categories
* Display confidence information
* Produce evaluation reports
* Visualize model performance
* Export trained models

---

## What the System Cannot Do

This project does not:

* Verify factual correctness
* Search the internet for evidence
* Replace professional fact-checkers
* Guarantee real-world accuracy
* Understand current events
* Make moderation decisions
* Determine objective truth

Predictions should be interpreted as stylistic similarity assessments rather than factual judgments.

---

## Model Pipeline

The workflow follows a standard machine learning pipeline:

```text
Input Text
    ↓
Text Cleaning
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Probability Estimation
    ↓
REAL / FAKE / UNCERTAIN
```

### TF-IDF Representation

The model converts textual content into numerical features using Term Frequency–Inverse Document Frequency (TF-IDF), allowing machine learning algorithms to identify important words and patterns.

### Logistic Regression

Logistic Regression serves as the classifier due to its efficiency, interpretability, and strong performance on sparse text data.

### Confidence-Based Classification

Instead of forcing every prediction into a binary decision, the project introduces an uncertainty zone:

```text
Low Probability      → REAL
Medium Probability   → UNCERTAIN
High Probability     → FAKE
```

This helps reduce overconfident predictions on ambiguous inputs.

---

## Web Dashboard

The Streamlit dashboard provides a simple interface for testing the model.

### Dashboard Capabilities

* Paste news text directly into the application
* View prediction results instantly
* Examine probability scores
* Review model metrics
* Understand confidence levels
* Read responsible-use guidance

### Example Output

```text
Prediction: UNCERTAIN
Fake-style Probability: 54.2%

The model is not sufficiently confident
to make a strong classification.
```

Run the dashboard:

```bash
streamlit run src/streamlit_app.py
```

---

## Project Structure

```text
project-root/
│
├── .github/
│   └── workflows/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── docs/
│   ├── data_statement.md
│   └── model_card.md
│
├── outputs/
│   ├── charts/
│   ├── metrics.json
│   ├── leakage_report.json
│   ├── model.joblib
│   └── pipeline.joblib
│
├── src/
│   ├── detect_fake_news.py
│   ├── streamlit_app.py
│   ├── text_clean.py
│   ├── train_model.py
│   └── utils.py
│
├── tests/
│
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── Makefile
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Create Virtual Environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Optional development tools:

```bash
pip install -r requirements-dev.txt
```

---

## Training

Train the model using:

```bash
python src/train_model.py
```

Training performs the following:

* Data loading
* Text preprocessing
* Feature extraction
* Model training
* Evaluation
* Metrics generation
* Visualization creation
* Artifact saving

Generated outputs:

```text
outputs/model.joblib
outputs/pipeline.joblib
outputs/metrics.json
outputs/leakage_report.json
outputs/charts/
```

---

## Running Predictions

### Command Line

```bash
python src/detect_fake_news.py --text "Example news text"
```

Example:

```bash
python src/detect_fake_news.py --text "Government announces new economic reforms."
```

JSON output:

```bash
python src/detect_fake_news.py --text "Example text" --json
```

---

## Prediction Labels

| Label     | Description                               |
| --------- | ----------------------------------------- |
| REAL      | Similar to real-news examples             |
| FAKE      | Similar to fake-news examples             |
| UNCERTAIN | Confidence too close to decision boundary |

Example:

```json
{
  "label": "UNCERTAIN",
  "fake_probability": 0.54,
  "confidence": "low"
}
```

---

## Evaluation Metrics

The project evaluates performance using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Cross Validation

Results are stored in:

```text
outputs/metrics.json
```

Visualization files are stored in:

```text
outputs/charts/
```

---

## Dataset Leakage Analysis

Machine learning models can sometimes learn shortcuts from datasets rather than meaningful patterns.

To improve transparency, this project includes a dedicated leakage analysis report:

```text
outputs/leakage_report.json
```

The report helps identify whether the model may be relying on dataset-specific artifacts rather than generalizable language patterns.

---

## Testing

Execute the test suite:

```bash
pytest
```

Tests cover:

* Text preprocessing
* Prediction pipeline
* Training workflow
* Utility functions
* Model compatibility

---

## Technology Stack

### Core Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

### Visualization

* Matplotlib

### Development Tools

* Pytest
* GitHub Actions

---

## Responsible Use

This project is intended for:

* Educational purposes
* NLP experimentation
* Machine learning portfolios
* Research demonstrations
* Software engineering practice

This project should not be used for:

* Automated fact-checking
* News verification
* Political moderation
* Legal decisions
* High-risk environments

Human review remains essential when evaluating news content.

---

## Future Enhancements

Potential improvements include:

* Transformer-based NLP models
* Explainable AI features
* SHAP visualizations
* Docker deployment
* Cloud hosting
* External dataset benchmarking
* Model calibration improvements
* Advanced feature engineering
* Real-time inference APIs

---

## License

This project is available for educational, research, and portfolio purposes.

Users are encouraged to maintain transparency regarding model limitations and avoid presenting the system as a factual truth verification tool.
