import pandas as pd
import datetime

def age(data: pd.DataFrame):
    current_year = datetime.datetime.now().year
    
    # Calculate age by subtracting the birth year from the current year
    return current_year - pd.to_datetime(data['dob']).dt.year

AGGREGATED_FEATURES = {
    "age": age,
}