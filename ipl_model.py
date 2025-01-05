import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def prepare_data(filepath='ipl_data.csv'):
    ipl_df = pd.read_csv(filepath)
    
    irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
    ipl_df = ipl_df.drop(irrelevant, axis=1)
    
    const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                  'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                  'Delhi Daredevils', 'Sunrisers Hyderabad']
    
    ipl_df = ipl_df[(ipl_df['bat_team'].isin(const_teams)) & 
                    (ipl_df['bowl_team'].isin(const_teams))]
    
    ipl_df = ipl_df[ipl_df['overs'] >= 5.0]
    
    le = LabelEncoder()
    ipl_df['bat_team'] = le.fit_transform(ipl_df['bat_team'])
    ipl_df['bowl_team'] = le.fit_transform(ipl_df['bowl_team'])
    
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1])], 
                                        remainder='passthrough')
    ipl_df = np.array(columnTransformer.fit_transform(ipl_df))
    
    cols = ['batting_team_' + team for team in const_teams] + \
           ['bowling_team_' + team for team in const_teams] + \
           ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']
    
    df = pd.DataFrame(ipl_df, columns=cols)
    return df, const_teams

def train_models(df):
    X = df.drop(['total'], axis=1)
    y = df['total']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standard scaling for PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Initialize models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(random_state=42)
    
    # Train base models
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    
    # Create ensemble predictions
    rf_pred = rf.predict(X_train)
    xgb_pred = xgb.predict(X_train)
    
    # Stack predictions
    stack_features = np.column_stack((rf_pred, xgb_pred))
    
    # Train meta-model
    meta_model = LinearRegression()
    meta_model.fit(stack_features, y_train)
    
    # Train PCA model
    pca_model = RandomForestRegressor(n_estimators=100, random_state=42)
    pca_model.fit(X_train_pca, y_train)
    
    # Store models in dictionary
    models = {
        'random_forest': rf,
        'xgboost': xgb,
        'stacking': {
            'rf': rf,
            'xgb': xgb,
            'meta': meta_model
        },
        'pca_rf': pca_model
    }
    
    # Evaluate models
    results = {}
    
    # Evaluate RF and XGB
    for name in ['random_forest', 'xgboost']:
        y_pred = models[name].predict(X_test)
        results[name] = calculate_metrics(y_test, y_pred)
    
    # Evaluate Stacking
    rf_test_pred = models['stacking']['rf'].predict(X_test)
    xgb_test_pred = models['stacking']['xgb'].predict(X_test)
    stack_test = np.column_stack((rf_test_pred, xgb_test_pred))
    stack_pred = models['stacking']['meta'].predict(stack_test)
    results['stacking'] = calculate_metrics(y_test, stack_pred)
    
    # Evaluate PCA
    pca_pred = models['pca_rf'].predict(X_test_pca)
    results['pca_rf'] = calculate_metrics(y_test, pca_pred)
    
    return models, pca, scaler, results

def calculate_metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

def save_models(models, pca, scaler, results):
    components = {
        'models': models,
        'pca': pca,
        'scaler': scaler,
        'results': results
    }
    with open('ipl_models.pkl', 'wb') as f:
        pickle.dump(components, f)

if __name__ == "__main__":
    print("Loading and preparing data...")
    df, teams = prepare_data()
    
    print("Training models...")
    models, pca, scaler, results = train_models(df)
    
    print("Saving models...")
    save_models(models, pca, scaler, results)
    
    print("\nModel Comparison:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R2 Score: {metrics['r2']:.2f}")