import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = []
        
    def engineer_features(self, data):
        """
        Advanced feature engineering for CTR/CVR prediction
        """
        print("Starting feature engineering...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # 1. Categorical encoding
        categorical_features = ['user_gender', 'user_location', 'user_device_type', 
                              'item_category', 'season']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
        
        # 2. Numerical feature scaling
        numerical_features = ['user_age', 'item_price', 'item_rating', 'item_review_count',
                            'user_session_length', 'user_page_views', 'user_previous_purchases',
                            'user_cart_size', 'user_lifetime_value', 'user_recency_days',
                            'user_frequency_score']
        
        for feature in numerical_features:
            if feature not in self.scalers:
                self.scalers[feature] = StandardScaler()
                df[f'{feature}_scaled'] = self.scalers[feature].fit_transform(df[[feature]])
            else:
                df[f'{feature}_scaled'] = self.scalers[feature].transform(df[[feature]])
        
        # 3. Binning features
        df['price_bin'] = pd.cut(df['item_price'], bins=10, labels=False)
        df['age_bin'] = pd.cut(df['user_age'], bins=5, labels=False)
        df['rating_bin'] = pd.cut(df['item_rating'], bins=5, labels=False)
        
        # 4. Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 5. Interaction features
        df['price_rating_interaction'] = df['item_price'] * df['item_rating']
        df['age_price_interaction'] = df['user_age'] * df['item_price']
        df['session_pageviews_ratio'] = df['user_session_length'] / (df['user_page_views'] + 1)
        
        # 6. User behavior features
        df['is_returning_user'] = (df['user_previous_purchases'] > 0).astype(int)
        df['is_high_value_user'] = (df['user_lifetime_value'] > df['user_lifetime_value'].quantile(0.8)).astype(int)
        df['is_frequent_user'] = (df['user_frequency_score'] > df['user_frequency_score'].quantile(0.7)).astype(int)
        
        # 7. Item popularity features
        item_click_rates = df.groupby('item_id')['click'].mean()
        item_conversion_rates = df.groupby('item_id')['conversion'].mean()
        df['item_historical_ctr'] = df['item_id'].map(item_click_rates)
        df['item_historical_cvr'] = df['item_id'].map(item_conversion_rates)
        
        # 8. User engagement features
        user_engagement = df.groupby('user_id').agg({
            'click': 'mean',
            'conversion': 'mean',
            'user_session_length': 'mean',
            'user_page_views': 'mean'
        }).add_suffix('_user_avg')
        
        df = df.merge(user_engagement, left_on='user_id', right_index=True, how='left')
        
        # 9. Category-specific features
        category_stats = df.groupby('item_category').agg({
            'item_price': ['mean', 'std'],
            'item_rating': 'mean',
            'click': 'mean'
        })
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.add_prefix('category_')
        
        df = df.merge(category_stats, left_on='item_category', right_index=True, how='left')
        
        # 10. Recency, Frequency, Monetary (RFM) features
        df['recency_score'] = 1 / (df['user_recency_days'] + 1)  # Higher score for recent users
        df['monetary_score'] = np.log1p(df['user_lifetime_value'])
        df['rfm_score'] = df['recency_score'] * df['user_frequency_score'] * df['monetary_score']
        
        print(f"Feature engineering complete. Shape: {df.shape}")
        
        # Store feature names for model training
        self.feature_names = [col for col in df.columns if col not in 
                            ['user_id', 'item_id', 'click', 'conversion', 'ctr_probability', 'cvr_probability']]
        
        return df
    
    def get_feature_matrix(self, df, target_type='ctr'):
        """
        Get feature matrix and target for model training
        """
        X = df[self.feature_names].fillna(0)
        
        if target_type == 'ctr':
            y = df['click']
        elif target_type == 'cvr':
            # For CVR, only use samples where click = 1
            mask = df['click'] == 1
            X = X[mask]
            y = df[mask]['conversion']
        else:
            raise ValueError("target_type must be 'ctr' or 'cvr'")
        
        return X, y
    
    def save_feature_config(self, filepath='feature_config.json'):
        """Save feature engineering configuration"""
        config = {
            'feature_names': self.feature_names,
            'label_encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()},
            'scaler_params': {k: {'mean': v.mean_.tolist(), 'scale': v.scale_.tolist()} 
                            for k, v in self.scalers.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Feature configuration saved to {filepath}")

if __name__ == "__main__":
    # Load a sample of the data for testing
    print("Loading sample data for feature engineering test...")
    
    # This would normally load from the generated chunks
    # For testing, we'll create a small sample
    np.random.seed(42)
    n_samples = 100000
    
    sample_data = pd.DataFrame({
        'user_id': np.random.randint(1, 10000, n_samples),
        'item_id': np.random.randint(1, 1000, n_samples),
        'user_age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'user_gender': np.random.choice(['M', 'F'], n_samples),
        'user_location': np.random.choice(['US', 'UK', 'CA'], n_samples),
        'user_device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
        'item_category': np.random.choice(['electronics', 'fashion', 'home'], n_samples),
        'item_price': np.random.lognormal(3, 1, n_samples),
        'item_rating': np.random.normal(4.2, 0.8, n_samples).clip(1, 5),
        'item_review_count': np.random.exponential(50, n_samples).astype(int),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'season': np.random.choice(['spring', 'summer', 'fall', 'winter'], n_samples),
        'user_session_length': np.random.exponential(15, n_samples),
        'user_page_views': np.random.poisson(8, n_samples),
        'user_previous_purchases': np.random.poisson(2, n_samples),
        'user_cart_size': np.random.poisson(1.5, n_samples),
        'user_lifetime_value': np.random.lognormal(5, 1, n_samples),
        'user_recency_days': np.random.exponential(30, n_samples),
        'user_frequency_score': np.random.gamma(2, 2, n_samples),
        'click': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'conversion': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    
    # Test feature engineering
    fe = FeatureEngineer()
    engineered_data = fe.engineer_features(sample_data)
    
    print(f"Original features: {len(sample_data.columns)}")
    print(f"Engineered features: {len(engineered_data.columns)}")
    print(f"Model features: {len(fe.feature_names)}")
    
    # Test feature matrix extraction
    X_ctr, y_ctr = fe.get_feature_matrix(engineered_data, 'ctr')
    X_cvr, y_cvr = fe.get_feature_matrix(engineered_data, 'cvr')
    
    print(f"CTR dataset shape: {X_ctr.shape}, {y_ctr.shape}")
    print(f"CVR dataset shape: {X_cvr.shape}, {y_cvr.shape}")
    
    # Save configuration
    fe.save_feature_config()
    
    print("Feature engineering test completed successfully!")
