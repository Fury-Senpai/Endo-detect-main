# EndoDetect Machine Learning Model

This project aims to develop a machine learning model to predict the likelihood of endometriosis based on various symptoms reported by patients. The model utilizes a dataset containing clinical features and diagnoses to train and evaluate its performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Training](#model-training)
- [Results](#results)
- [Future Improvements]

## Installation

To run this project, you need to have Python installed along the libraries mentioned in requirements.txt file.


## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/endometriosis-detection.git
   cd endometriosis-detection
   ```

2. Prepare your dataset in CSV format and place it in the `data` directory. The dataset should contain the following columns:
   - 'Label'
   - `Irregular / Missed periods`
   - `Cramping`
   - `Menstrual clots`
   - `Infertility`
   - `Pain / Chronic pain`
   - `Diarrhea`
   - `Long menstruation`
   - `Vomiting / constant vomiting`
   - `Migraines`
   - `Extreme Bloating`
   - `Leg pain`
   - `Depression`
   - `Fertility Issues`
   - `Ovarian cysts`
   - `Painful urination`
   - `Pain after Intercourse`
   - `Digestive / GI problems`
   - `Anaemia / Iron deficiency`
   - `Hip pain`
   - `Vaginal Pain/Pressure`
   - `Cysts (unspecified)`
   - `Abnormal uterine bleeding`
   - `Hormonal problems`
   - `Feeling sick`
   - `Abdominal Cramps during Intercourse`
   - `Insomnia / Sleeplessness`
   - `Loss of appetite`
   - `Diagnosis` (0 for No Endometriosis, 1 for Endometriosis)

3. Run the model training script:

   ```bash
   python src/model_training.py
   ```

4. To make predictions based on user input, run:

   ```bash
   python src/predict_user_input.py
   ```

## Data Description

The dataset used for training the model consists of clinical features related to endometriosis symptoms. The target variable is the `Diagnosis`, which indicates whether a patient has endometriosis (1) or not (0).

## Model Training

The model is trained using the following steps:

1. **Data Preprocessing**: Load and preprocess the data, handling missing values and splitting features and target variables.
2. **Model Selection**: The model used is XGBoost, which is known for its performance in classification tasks.
3. **Hyperparameter Tuning**: Grid search is employed to find the best hyperparameters for the model.
4. **Class Imbalance Handling**: SMOTE is used to address class imbalance in the dataset.

## Results

After training the model, the following results were obtained:

- **Accuracy**: 74%
- **Confusion Matrix**:
  ```
  [[77 24]
   [22 55]]
  ```
- **Classification Report**:
  ```
              precision    recall  f1-score   support

           0       0.78      0.76      0.77       101
           1       0.70      0.71      0.71        77

    accuracy                           0.74       178
   macro avg       0.74      0.74      0.74       178
weighted avg       0.74      0.74      0.74       178
  ``


# EndoDetect Website

EndoDetect is a simple, responsive website that connects users with an early detection model for Endometriosis. This project combines a heartfelt user interface with seamless backend integration to provide symptom-based predictions.


## Project Structure

The website is divided into two major parts:

###  Frontend
- Built with **HTML**, **CSS**, **JavaScript**, and **Bootstrap**
- Pages include:
  - `home.html` – Overview
  - `index.html` – Symptom input form
  - `creator.html` – Project story and creator section
  - `contact.html` – Contact form

### Backend
- Developed with **Flask**
- Receives form input from `index.html`
- Sends it to the trained ML model (`trained_model.pkl`)
- Returns prediction to be displayed to the user


## 🚀 How to Run Locally

1. Clone this repo and navigate to the backend folder:

2. Install the required packages

3. Run the Flask app:
python app.py


4. Open `frontend/home.html` in your browser to use the site.


##  Features

-  Clean and responsive UI
-  Connects to trained model via Flask API
-  Displays prediction result instantly
-  Emotionally meaningful experience
-  Easy to deploy and modify



##  Key Libraries Used

- Flask  
- Flask-CORS  
- Requests (for API calls)  
- HTML / CSS / JS  
- Bootstrap  


## Future Improvements

To improve the model's accuracy above 80%, consider the following strategies:

- Enhance feature engineering by creating new features or transforming existing ones.
- Experiment with different algorithms such as Random Forest or Neural Networks.
- Implement more advanced hyperparameter tuning techniques.
- Increase the dataset size or use data augmentation techniques.
- Explore ensemble methods to combine predictions from multiple models.


##  Creator

Made with love by **Pawni Dixit**

---

## 🌸 Final Note

This website isn't just a tool — it's a quiet revolution in support, empathy, and awareness.  
If it helps even one person get closer to an answer, it’s worth it.


