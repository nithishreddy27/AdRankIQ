import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import json

def generate_large_scale_dataset(n_samples=10_000_000):
    """
    Generate a large-scale dataset for CTR and CVR prediction
    Features include user demographics, item features, contextual features, and interaction history
    """
    print(f"Generating {n_samples:,} samples for CTR/CVR prediction...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate user features
    print("Generating user features...")
    user_ids = np.random.randint(1, 1_000_000, n_samples)
    user_ages = np.random.normal(35, 12, n_samples).clip(18, 80).astype(int)
    user_genders = np.random.choice(['M', 'F'], n_samples, p=[0.52, 0.48])
    user_locations = np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'IN'], n_samples, 
                                    p=[0.3, 0.15, 0.1, 0.08, 0.08, 0.07, 0.07, 0.15])
    user_device_types = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.65, 0.25, 0.1])
    
    # Generate item features
    print("Generating item features...")
    item_ids = np.random.randint(1, 100_000, n_samples)
    item_categories = np.random.choice(['electronics', 'fashion', 'home', 'books', 'sports', 'beauty'], 
                                     n_samples, p=[0.2, 0.25, 0.15, 0.1, 0.15, 0.15])
    item_prices = np.random.lognormal(3, 1, n_samples).clip(1, 1000)
    item_ratings = np.random.normal(4.2, 0.8, n_samples).clip(1, 5)
    item_review_counts = np.random.exponential(50, n_samples).astype(int)
    
    # Generate contextual features
    print("Generating contextual features...")
    hours = np.random.randint(0, 24, n_samples)
    days_of_week = np.random.randint(0, 7, n_samples)
    is_weekend = (days_of_week >= 5).astype(int)
    seasons = np.random.choice(['spring', 'summer', 'fall', 'winter'], n_samples)
    
    # Generate user behavior features
    print("Generating user behavior features...")
    user_session_lengths = np.random.exponential(15, n_samples)  # minutes
    user_page_views = np.random.poisson(8, n_samples)
    user_previous_purchases = np.random.poisson(2, n_samples)
    user_cart_size = np.random.poisson(1.5, n_samples)
    
    # Generate advanced features
    user_lifetime_value = np.random.lognormal(5, 1, n_samples)
    user_recency_days = np.random.exponential(30, n_samples)
    user_frequency_score = np.random.gamma(2, 2, n_samples)
    
    # Create feature interactions
    price_age_interaction = item_prices * (user_ages / 100)
    category_gender_match = ((item_categories == 'fashion') & (user_genders == 'F')).astype(int) + \
                           ((item_categories == 'electronics') & (user_genders == 'M')).astype(int)
    
    # Generate CTR labels (click-through rate)
    print("Generating CTR labels...")
    # CTR is influenced by multiple factors
    ctr_logits = (
        -2.5 +  # base rate
        0.3 * (item_ratings - 3) +  # higher rated items get more clicks
        0.2 * np.log(item_review_counts + 1) +  # more reviews = more clicks
        -0.1 * np.log(item_prices) +  # cheaper items get more clicks
        0.4 * category_gender_match +  # gender-category match
        0.2 * (user_previous_purchases > 0) +  # returning customers click more
        0.1 * is_weekend +  # weekend effect
        -0.05 * np.abs(hours - 14) +  # peak at 2 PM
        0.15 * (user_device_types == 'mobile') +  # mobile users click more
        np.random.normal(0, 0.3, n_samples)  # noise
    )
    
    ctr_probabilities = 1 / (1 + np.exp(-ctr_logits))
    clicks = np.random.binomial(1, ctr_probabilities)
    
    # Generate CVR labels (conversion rate) - only for clicked items
    print("Generating CVR labels...")
    cvr_logits = np.where(clicks == 1,
        -1.8 +  # base conversion rate
        0.4 * (item_ratings - 3) +
        -0.3 * np.log(item_prices) +  # price sensitivity for conversion
        0.3 * (user_previous_purchases > 0) +
        0.2 * (user_cart_size > 0) +
        0.1 * (user_lifetime_value > np.median(user_lifetime_value)) +
        -0.1 * user_recency_days / 30 +  # recent users convert better
        0.2 * user_frequency_score +
        np.random.normal(0, 0.2, n_samples),
        0  # no conversion if no click
    )
    
    cvr_probabilities = np.where(clicks == 1, 1 / (1 + np.exp(-cvr_logits)), 0)
    conversions = np.where(clicks == 1, np.random.binomial(1, cvr_probabilities), 0)
    
    # Create DataFrame
    print("Creating DataFrame...")
    data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'user_age': user_ages,
        'user_gender': user_genders,
        'user_location': user_locations,
        'user_device_type': user_device_types,
        'item_category': item_categories,
        'item_price': item_prices,
        'item_rating': item_ratings,
        'item_review_count': item_review_counts,
        'hour': hours,
        'day_of_week': days_of_week,
        'is_weekend': is_weekend,
        'season': seasons,
        'user_session_length': user_session_lengths,
        'user_page_views': user_page_views,
        'user_previous_purchases': user_previous_purchases,
        'user_cart_size': user_cart_size,
        'user_lifetime_value': user_lifetime_value,
        'user_recency_days': user_recency_days,
        'user_frequency_score': user_frequency_score,
        'price_age_interaction': price_age_interaction,
        'category_gender_match': category_gender_match,
        'click': clicks,
        'conversion': conversions,
        'ctr_probability': ctr_probabilities,
        'cvr_probability': cvr_probabilities
    })
    
    # Add some derived features
    data['conversion_value'] = np.where(data['conversion'] == 1, 
                                      data['item_price'] * np.random.uniform(0.8, 1.2, n_samples), 0)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {data.shape}")
    print(f"CTR: {data['click'].mean():.4f}")
    print(f"CVR (overall): {data['conversion'].mean():.4f}")
    print(f"CVR (given click): {data[data['click']==1]['conversion'].mean():.4f}")
    
    return data

def save_dataset_chunks(data, chunk_size=1_000_000):
    """Save dataset in chunks for memory efficiency"""
    print("Saving dataset in chunks...")
    
    n_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        chunk = data.iloc[start_idx:end_idx]
        
        filename = f"data_chunk_{i+1}.parquet"
        chunk.to_parquet(filename, index=False)
        print(f"Saved {filename}: {len(chunk):,} samples")
    
    # Save metadata
    metadata = {
        'total_samples': len(data),
        'n_chunks': n_chunks,
        'chunk_size': chunk_size,
        'features': list(data.columns),
        'ctr_mean': float(data['click'].mean()),
        'cvr_mean': float(data['conversion'].mean()),
        'cvr_given_click': float(data[data['click']==1]['conversion'].mean()) if data['click'].sum() > 0 else 0
    }
    
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Dataset metadata saved!")
    return metadata

if __name__ == "__main__":
    # Generate the dataset
    dataset = generate_large_scale_dataset(10_000_000)
    
    # Save in chunks
    metadata = save_dataset_chunks(dataset)
    
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total samples: {metadata['total_samples']:,}")
    print(f"Overall CTR: {metadata['ctr_mean']:.4f}")
    print(f"Overall CVR: {metadata['cvr_mean']:.4f}")
    print(f"CVR given click: {metadata['cvr_given_click']:.4f}")
