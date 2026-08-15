import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def build_feature_row(inputs: dict, feature_cols: list) -> pd.DataFrame:
    """
    inputs: raw form values from the app
    feature_cols: the exact training column order (loaded from feature_columns.pkl)
    Returns a single-row DataFrame matching training format exactly.
    """
    distance = haversine_distance(
        inputs['rest_lat'], inputs['rest_lon'],
        inputs['deliv_lat'], inputs['deliv_lon']
    )

    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}

    row = {
        'Delivery_person_Age': inputs['age'],
        'Delivery_person_Ratings': inputs['ratings'],
        'Vehicle_condition': inputs['vehicle_condition'],
        'multiple_deliveries': inputs['multiple_deliveries'],
        'Festival': 1 if inputs['festival'] == 'Yes' else 0,
        'Road_traffic_density': traffic_map[inputs['traffic']],
        'Distance_km': distance,
        'Order_Hour': inputs['order_hour'],
        'Is_Rush_Hour': 1 if inputs['order_hour'] in [8, 9, 12, 13, 19, 20, 21] else 0,
    }

    # one-hot columns: set the matching dummy column to 1, everything else defaults to 0
    for col in feature_cols:
        if col.startswith('Weather_conditions_') and col == f"Weather_conditions_{inputs['weather']}":
            row[col] = 1
        elif col.startswith('Type_of_order_') and col == f"Type_of_order_{inputs['order_type']}":
            row[col] = 1
        elif col.startswith('Type_of_vehicle_') and col == f"Type_of_vehicle_{inputs['vehicle_type']}":
            row[col] = 1
        elif col.startswith('City_') and col == f"City_{inputs['city']}":
            row[col] = 1

    # this is the critical step - reindex guarantees the column order matches training exactly
    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=feature_cols, fill_value=0)
    return df_row