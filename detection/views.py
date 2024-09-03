# detection/views.py

import numpy as np
import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import csv

def index(request):
    return render(request, 'detection/index.html')

def predict(request):
    if request.method == 'POST':
        # Retrieve data from form
        features = [
            float(request.POST['radius_mean']),
            float(request.POST['texture_mean']),
            float(request.POST['smoothness_mean']),
            float(request.POST['compactness_mean']),
            float(request.POST['symmetry_mean']),
            float(request.POST['fractal_dimension_mean']),
            float(request.POST['radius_se']),
            float(request.POST['texture_se']),
            float(request.POST['smoothness_se']),
            float(request.POST['compactness_se'])
        ]
        
        scaler = joblib.load('C:/Users/nunna/scaler.pkl')  # Load the saved scaler
        clf = TabNetClassifier()
        clf.load_model('C:/Users/nunna/tabnet_classifier.zip')

        # Scale the input features
        scaled_features = scaler.transform([features])
        
        # Make a prediction
        prediction = clf.predict(scaled_features)
        
        # Interpret the result
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        
        # Return the result as JSON
        return JsonResponse({'prediction': result})
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def predict_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)

        # Ensure the feature columns match those used during training
        feature_names = [
            'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
            'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'smoothness_se',
            'compactness_se'
        ]

        # Check if the CSV file contains all the expected features
        if not all(feature in df.columns for feature in feature_names):
            return JsonResponse({'error': f"CSV file is missing some of the expected feature columns: {feature_names}"}, status=400)

        # Extract features and scale them
        X_new = df[feature_names].values
        scaler = joblib.load('C:/Users/nunna/scaler.pkl')  # Load the saved scaler
        clf = TabNetClassifier()
        clf.load_model('C:/Users/nunna/tabnet_classifier.zip')  # Load the saved model

        # Scale the input features
        X_new_scaled = scaler.transform(X_new)

        # Make predictions
        predictions = clf.predict(X_new_scaled)

        # Add predictions to the original dataframe
        df['prediction'] = ['Malignant' if pred == 1 else 'Benign' for pred in predictions]

        # Save the results to a new CSV file
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="Predictions.csv"'
        df.to_csv(path_or_buf=response, index=False)
        
        # Debugging information
        print("CSV file generated and ready for download.")
        
        return response

    return JsonResponse({'error': 'Invalid request method or no file uploaded'}, status=400)