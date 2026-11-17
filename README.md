# Crop Recommendation using Random Forest

A Flask web app that recommends which crop to grow, based on soil nutrients (N, P, K), soil pH and rainfall.

> **Academic project:** developed as part of my Master of Computer Applications (MCA), Annamacharya Group of Institutions.

![Home page](images/home.png)

## Problem
Choosing a crop that suits the soil and rainfall conditions of a farm improves yield. This project trains a Random Forest classifier on a public crop dataset and serves it through a simple web form.

## Dataset
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) (Kaggle): 2,200 rows, 22 crops (100 samples each), with N, P, K, temperature, humidity, pH and rainfall.

## Approach
- **Features used:** N, P, K, pH, rainfall, plus two engineered features (N+P and N+K). Temperature and humidity are in the dataset but are not used by the model, so the web form does not ask for them.
- **Model:** `RandomForestClassifier` (100 trees), stratified 80/20 train/test split.
- **Result:** about **97% accuracy** on the held-out test set (440 rows). Run `python train_model.py` to reproduce it.
- **Serving:** Flask app loads the saved model and predicts from the form values. Invalid input returns an error page instead of crashing.

![Prediction form](images/form.png)
![Prediction result](images/result.png)

## Run it locally
```bash
git clone https://github.com/Pooja012/Crop-Recommendation-System.git
cd Crop-Recommendation-System
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash. Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

python train_model.py             # trains and saves models/RandomForest.pkl
python app.py                     # open http://127.0.0.1:5000
```
Try N=90, P=42, K=43, pH=6.5, rainfall=203, which should recommend **rice**.

## Project structure
```
├── Data/                  crop_recommendation.csv
├── models/                saved Random Forest (created by train_model.py, not committed)
├── static/ , templates/   CSS, images and HTML pages
├── images/                screenshots used in this README
├── train_model.py         trains and saves the model
├── app.py                 Flask web app
└── requirements.txt
```
## My contribution
- Trained and evaluated the Random Forest model on the Crop Recommendation dataset.
- Engineered features (N+P, N+K) and built the Flask web interface for predictions.
- Cleaned up the project for reproducibility: retraining script, input validation, requirements file, and documentation.


## Limitations and next steps
- The state/city dropdowns on the form are not used by the model.
- The dataset is a public, mostly synthetic-looking benchmark, so the high accuracy shows the method works on this data and does not guarantee real-farm performance.
- Next: add temperature and humidity (for example from a weather API), show the top 3 crops with probabilities, and deploy it.

## Credits
- Dataset: Atharva Ingle, Kaggle.
- Web UI: adapted from [add the source you started from, if any].

## Author
**Pooja Batta** · [LinkedIn](https://www.linkedin.com/in/pooja-batta-9b4184254/) · [GitHub](https://github.com/Pooja012)
