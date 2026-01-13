import joblib
import pandas as pd
import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# Load XGBoost model and preprocessing artifacts
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

def create_df(date, hour, latitude, longitude, place, age, race, gender, precinct, borough):
    """
    Create a DataFrame with all features needed for prediction
    based on your model's feature engineering
    """
    
    # Extract date components
    year = date.year
    month = date.month
    day = date.day
    hour = int(hour) if int(hour) < 24 else 0
    
    # Map weekday
    weekday = date.strftime('%A')  # e.g., 'Monday', 'Tuesday', etc.
    
    # Basic location features
    ADDR_PCT_CD = float(precinct) if precinct else 0.0
    JURISDICTION_CODE = 0  # Default to NYPD
    
    # Map borough
    boro = borough.upper() if borough else 'UNKNOWN'
    
    # Map place to premise type
    if place == "In park":
        PREM_TYP_DESC = "PARK/PLAYGROUND"
        OCCURENCE = "INSIDE"
    elif place == "In public housing":
        PREM_TYP_DESC = "RESIDENCE - PUBLIC HOUSING"
        OCCURENCE = "INSIDE"
    elif place == "In station":
        PREM_TYP_DESC = "TRANSIT - NYC SUBWAY"
        OCCURENCE = "INSIDE"
    else:
        PREM_TYP_DESC = "STREET"
        OCCURENCE = "FRONT OF"
    
    # Determine age group
    if age < 18:
        VIC_AGE_GROUP = "<18"
    elif 18 <= age <= 24:
        VIC_AGE_GROUP = "18-24"
    elif 25 <= age <= 44:
        VIC_AGE_GROUP = "25-44"
    elif 45 <= age <= 64:
        VIC_AGE_GROUP = "45-64"
    else:
        VIC_AGE_GROUP = "65+"
    
    # Map gender
    VIC_SEX = 'M' if gender == "Male" else 'F'
    
    # Map race
    VIC_RACE = race.upper()
    
    # Additional features from your feature engineering
    is_weekend = 1 if weekday in ['Saturday', 'Sunday'] else 0
    is_night = 1 if (hour >= 20 or hour <= 6) else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    
    # Season
    if month in [12, 1, 2]:
        season = 'Winter'
    elif month in [3, 4, 5]:
        season = 'Spring'
    elif month in [6, 7, 8]:
        season = 'Summer'
    else:
        season = 'Fall'
    
    # Default values for other required features
    COMPLETED = "COMPLETED"
    CRIME_CLASS = "FELONY"  # Default
    JURIS_DESC = "N.Y. POLICE DEPT"
    SUSP_AGE_GROUP = "UNKNOWN"
    SUSP_RACE = "UNKNOWN"
    SUSP_SEX = "(null)"
    
    # Location crime density (simplified - you'd calculate this from historical data)
    location_crime_density = 100  # Placeholder value
    
    # Create initial DataFrame with categorical values
    data_dict = {
        'year': year,
        'month': month,
        'day': day,
        'hour': hour,
        'Latitude': latitude,
        'Longitude': longitude,
        'ADDR_PCT_CD': ADDR_PCT_CD,
        'JURISDICTION_CODE': JURISDICTION_CODE,
        'weekday': weekday,
        'COMPLETED': COMPLETED,
        'CRIME_CLASS': CRIME_CLASS,
        'BORO_NM': boro,
        'PREM_TYP_DESC': PREM_TYP_DESC,
        'OCCURENCE': OCCURENCE,
        'SUSP_AGE_GROUP': SUSP_AGE_GROUP,
        'SUSP_RACE': SUSP_RACE,
        'SUSP_SEX': SUSP_SEX,
        'VIC_AGE_GROUP': VIC_AGE_GROUP,
        'VIC_RACE': VIC_RACE,
        'VIC_SEX': VIC_SEX,
        'season': season,
        'is_weekend': is_weekend,
        'is_night': is_night,
        'is_rush_hour': is_rush_hour,
        'location_crime_density': location_crime_density
    }
    
    # Encode categorical variables using your saved label encoders
    encoded_features = {}
    categorical_cols = ['weekday', 'COMPLETED', 'CRIME_CLASS', 'BORO_NM', 
                       'PREM_TYP_DESC', 'OCCURENCE', 'JURIS_DESC',
                       'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX',
                       'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX', 'season']
    
    for col in categorical_cols:
        if col in data_dict:
            try:
                # Transform using saved encoder
                encoded_value = label_encoders[col].transform([str(data_dict[col])])[0]
                encoded_features[col + '_encoded'] = encoded_value
            except:
                # If value not in encoder, use 0
                encoded_features[col + '_encoded'] = 0
    
    # Create final feature vector in the correct order
    feature_cols = ['year', 'month', 'day', 'hour', 'Latitude', 'Longitude',
                    'ADDR_PCT_CD', 'JURISDICTION_CODE',
                    'weekday_encoded', 'COMPLETED_encoded', 'CRIME_CLASS_encoded',
                    'BORO_NM_encoded', 'PREM_TYP_DESC_encoded', 'OCCURENCE_encoded',
                    'SUSP_AGE_GROUP_encoded', 'SUSP_RACE_encoded', 'SUSP_SEX_encoded',
                    'VIC_AGE_GROUP_encoded', 'VIC_RACE_encoded', 'VIC_SEX_encoded',
                    'season_encoded', 'is_weekend', 'is_night', 'is_rush_hour',
                    'location_crime_density']
    
    # Build final row
    final_data = []
    for col in feature_cols:
        if col in data_dict:
            final_data.append(data_dict[col])
        elif col in encoded_features:
            final_data.append(encoded_features[col])
        else:
            final_data.append(0)  # Default value
    
    # Create DataFrame
    df = pd.DataFrame([final_data], columns=feature_cols)
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled

def predict(data):
    """
    Make prediction and return crime category
    """
    # Get prediction (encoded value)
    pred_encoded = model.predict(data)[0]
    
    # Decode prediction to get actual crime category
    crime_category = label_encoders['CRIME_CATEGORY'].inverse_transform([pred_encoded])[0]
    
    # Map to specific crime types
    crime_details = {
        'DRUGS/ALCOHOL': [
            'DANGEROUS DRUGS', 
            'INTOXICATED & IMPAIRED DRIVING',
            'ALCOHOLIC BEVERAGE CONTROL LAW', 
            'UNDER THE INFLUENCE OF DRUGS', 
            'LOITERING FOR DRUG PURPOSES'
        ],
        'PROPERTY': [
            'BURGLARY', 
            'PETIT LARCENY', 
            'GRAND LARCENY', 
            'ROBBERY', 
            'THEFT-FRAUD', 
            'GRAND LARCENY OF MOTOR VEHICLE', 
            'FORGERY', 
            'ARSON',
            'POSSESSION OF STOLEN PROPERTY',
            'CRIMINAL MISCHIEF & RELATED OF'
        ],
        'PERSONAL': [
            'ASSAULT 3 & RELATED OFFENSES', 
            'FELONY ASSAULT',
            'OFFENSES AGAINST THE PERSON', 
            'HOMICIDE-NEGLIGENT,UNCLASSIFIE',
            'KIDNAPPING & RELATED OFFENSES',
            'DANGEROUS WEAPONS'
        ],
        'SEXUAL': [
            'SEX CRIMES', 
            'HARRASSMENT 2', 
            'RAPE', 
            'PROSTITUTION & RELATED OFFENSES',
            'FELONY SEX CRIMES'
        ]
    }
    
    # Return crime category and specific crimes
    crimes = crime_details.get(crime_category, ['UNKNOWN CRIME TYPE'])
    
    return crime_category, crimes

def get_prediction_probability(data):
    """
    Get prediction probabilities for all crime categories
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data)[0]
        crime_categories = label_encoders['CRIME_CATEGORY'].classes_
        
        # Create dictionary of probabilities
        prob_dict = {category: prob for category, prob in zip(crime_categories, probabilities)}
        return prob_dict
    return None