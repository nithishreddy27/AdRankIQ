import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RankingConfig:
    """Configuration for ranking system"""
    ctr_weight: float = 0.4
    cvr_weight: float = 0.6
    revenue_weight: float = 1.0
    diversity_weight: float = 0.1
    freshness_weight: float = 0.05
    popularity_weight: float = 0.1
    
    # Business constraints
    min_ctr_threshold: float = 0.01
    min_cvr_threshold: float = 0.005
    max_price_threshold: float = 1000.0
    
    # Ranking parameters
    top_k: int = 100
    diversification_window: int = 10
    
class ModelEnsemble:
    """Ensemble of CTR and CVR models for prediction"""
    
    def __init__(self):
        self.ctr_xgb_model = None
        self.ctr_nn_model = None
        self.cvr_xgb_model = None
        self.cvr_nn_model = None
        self.multitask_model = None
        self.feature_engineer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self, model_paths: Dict[str, str]):
        """Load all trained models"""
        print("Loading trained models...")
        
        # Load feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Load XGBoost models
        if 'ctr_xgb' in model_paths:
            self.ctr_xgb_model = xgb.Booster()
            self.ctr_xgb_model.load_model(model_paths['ctr_xgb'])
            print("CTR XGBoost model loaded")
            
        if 'cvr_xgb' in model_paths:
            self.cvr_xgb_model = xgb.Booster()
            self.cvr_xgb_model.load_model(model_paths['cvr_xgb'])
            print("CVR XGBoost model loaded")
        
        # Load Neural Network models would require model architecture
        # For demo purposes, we'll simulate predictions
        print("Model ensemble loaded successfully!")
        
    def predict_ctr(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CTR using ensemble of models"""
        predictions = []
        
        if self.ctr_xgb_model is not None:
            dtest = xgb.DMatrix(X)
            xgb_pred = self.ctr_xgb_model.predict(dtest)
            predictions.append(xgb_pred)
        
        # Simulate neural network predictions
        # In practice, you would load and use the actual trained models
        nn_pred = self._simulate_ctr_predictions(X)
        predictions.append(nn_pred)
        
        # Ensemble average
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return np.random.uniform(0.05, 0.25, len(X))  # Fallback
    
    def predict_cvr(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CVR using ensemble of models"""
        predictions = []
        
        if self.cvr_xgb_model is not None:
            dtest = xgb.DMatrix(X)
            xgb_pred = self.cvr_xgb_model.predict(dtest)
            predictions.append(xgb_pred)
        
        # Simulate neural network predictions
        nn_pred = self._simulate_cvr_predictions(X)
        predictions.append(nn_pred)
        
        # Ensemble average
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return np.random.uniform(0.01, 0.15, len(X))  # Fallback
    
    def _simulate_ctr_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Simulate CTR predictions based on features"""
        # This simulates what a trained neural network would predict
        base_ctr = 0.12
        
        # Feature-based adjustments
        adjustments = 0
        if 'item_rating_scaled' in X.columns:
            adjustments += 0.05 * X['item_rating_scaled']
        if 'item_price_scaled' in X.columns:
            adjustments -= 0.03 * X['item_price_scaled']
        if 'user_previous_purchases' in X.columns:
            adjustments += 0.02 * (X['user_previous_purchases'] > 0)
        
        ctr_logits = np.log(base_ctr / (1 - base_ctr)) + adjustments + np.random.normal(0, 0.1, len(X))
        return 1 / (1 + np.exp(-ctr_logits))
    
    def _simulate_cvr_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Simulate CVR predictions based on features"""
        # This simulates what a trained neural network would predict
        base_cvr = 0.08
        
        # Feature-based adjustments
        adjustments = 0
        if 'item_rating_scaled' in X.columns:
            adjustments += 0.08 * X['item_rating_scaled']
        if 'item_price_scaled' in X.columns:
            adjustments -= 0.06 * X['item_price_scaled']
        if 'user_lifetime_value_scaled' in X.columns:
            adjustments += 0.04 * X['user_lifetime_value_scaled']
        
        cvr_logits = np.log(base_cvr / (1 - base_cvr)) + adjustments + np.random.normal(0, 0.15, len(X))
        return 1 / (1 + np.exp(-cvr_logits))

class RankingSystem:
    """Advanced ranking system that balances relevance and monetization"""
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.model_ensemble = ModelEnsemble()
        self.feature_engineer = FeatureEngineer()
        
    def load_models(self, model_paths: Dict[str, str]):
        """Load trained models"""
        self.model_ensemble.load_models(model_paths)
        
    def rank_items(self, user_features: Dict, candidate_items: List[Dict], 
                   context_features: Dict = None) -> List[Dict]:
        """
        Rank items for a user balancing relevance and monetization
        
        Args:
            user_features: User profile features
            candidate_items: List of candidate items to rank
            context_features: Contextual features (time, location, etc.)
            
        Returns:
            List of ranked items with scores
        """
        print(f"Ranking {len(candidate_items)} items for user...")
        
        # Prepare feature matrix
        feature_data = self._prepare_features(user_features, candidate_items, context_features)
        
        # Engineer features
        engineered_features = self.feature_engineer.engineer_features(feature_data)
        X, _ = self.feature_engineer.get_feature_matrix(engineered_features, 'ctr')
        
        # Get predictions
        ctr_predictions = self.model_ensemble.predict_ctr(X)
        cvr_predictions = self.model_ensemble.predict_cvr(X)
        
        # Calculate ranking scores
        ranking_scores = self._calculate_ranking_scores(
            feature_data, ctr_predictions, cvr_predictions
        )
        
        # Apply business constraints
        valid_items = self._apply_business_constraints(feature_data, ranking_scores)
        
        # Diversification
        final_rankings = self._apply_diversification(valid_items)
        
        # Format results
        results = self._format_results(final_rankings, feature_data, 
                                     ctr_predictions, cvr_predictions)
        
        print(f"Ranking completed. Top item score: {results[0]['final_score']:.4f}")
        return results[:self.config.top_k]
    
    def _prepare_features(self, user_features: Dict, candidate_items: List[Dict], 
                         context_features: Dict = None) -> pd.DataFrame:
        """Prepare feature matrix for ranking"""
        
        # Create feature combinations for each user-item pair
        features_list = []
        
        for item in candidate_items:
            feature_row = {}
            
            # User features
            feature_row.update({f'user_{k}': v for k, v in user_features.items()})
            
            # Item features
            feature_row.update({f'item_{k}': v for k, v in item.items()})
            
            # Context features
            if context_features:
                feature_row.update(context_features)
            
            # Add some derived features
            feature_row['price_age_interaction'] = item.get('price', 0) * user_features.get('age', 35)
            feature_row['category_gender_match'] = self._calculate_category_gender_match(
                item.get('category', ''), user_features.get('gender', 'M')
            )
            
            features_list.append(feature_row)
        
        return pd.DataFrame(features_list)
    
    def _calculate_category_gender_match(self, category: str, gender: str) -> int:
        """Calculate category-gender matching score"""
        matches = {
            ('fashion', 'F'): 1,
            ('electronics', 'M'): 1,
            ('beauty', 'F'): 1,
            ('sports', 'M'): 1
        }
        return matches.get((category.lower(), gender.upper()), 0)
    
    def _calculate_ranking_scores(self, feature_data: pd.DataFrame, 
                                ctr_predictions: np.ndarray, 
                                cvr_predictions: np.ndarray) -> np.ndarray:
        """Calculate comprehensive ranking scores"""
        
        # Base relevance score (CTR * CVR weighted)
        relevance_score = (
            self.config.ctr_weight * ctr_predictions + 
            self.config.cvr_weight * cvr_predictions
        )
        
        # Revenue potential (CVR * Price)
        prices = feature_data.get('item_price', pd.Series([50] * len(feature_data)))
        revenue_score = cvr_predictions * prices * self.config.revenue_weight
        
        # Popularity score (based on review count)
        review_counts = feature_data.get('item_review_count', pd.Series([10] * len(feature_data)))
        popularity_score = np.log1p(review_counts) * self.config.popularity_weight
        
        # Freshness score (newer items get boost)
        # Simulate item age - in practice this would come from item features
        item_ages = np.random.exponential(30, len(feature_data))  # days
        freshness_score = np.exp(-item_ages / 90) * self.config.freshness_weight
        
        # Combine scores
        total_score = relevance_score + revenue_score + popularity_score + freshness_score
        
        # Normalize to 0-1 range
        if total_score.max() > total_score.min():
            total_score = (total_score - total_score.min()) / (total_score.max() - total_score.min())
        
        return total_score
    
    def _apply_business_constraints(self, feature_data: pd.DataFrame, 
                                  scores: np.ndarray) -> List[Tuple[int, float]]:
        """Apply business constraints and filter items"""
        
        valid_items = []
        
        for idx, score in enumerate(scores):
            # Get item features
            price = feature_data.iloc[idx].get('item_price', 0)
            
            # Apply constraints
            if price > self.config.max_price_threshold:
                continue
                
            # Add more business logic here
            # e.g., inventory checks, category restrictions, etc.
            
            valid_items.append((idx, score))
        
        # Sort by score
        valid_items.sort(key=lambda x: x[1], reverse=True)
        
        return valid_items
    
    def _apply_diversification(self, ranked_items: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Apply diversification to avoid showing too many similar items"""
        
        if len(ranked_items) <= self.config.diversification_window:
            return ranked_items
        
        diversified = []
        used_categories = set()
        
        # First pass: take top items with category diversity
        for idx, score in ranked_items:
            if len(diversified) >= self.config.top_k:
                break
                
            # In practice, you'd get category from feature_data
            # For demo, we'll simulate category diversity
            category = f"category_{idx % 5}"  # Simulate 5 categories
            
            if len(diversified) < self.config.diversification_window:
                diversified.append((idx, score))
                used_categories.add(category)
            elif category not in used_categories:
                diversified.append((idx, score))
                used_categories.add(category)
        
        # Second pass: fill remaining slots with highest scoring items
        remaining_slots = self.config.top_k - len(diversified)
        used_indices = {idx for idx, _ in diversified}
        
        for idx, score in ranked_items:
            if remaining_slots <= 0:
                break
            if idx not in used_indices:
                diversified.append((idx, score))
                remaining_slots -= 1
        
        return diversified
    
    def _format_results(self, ranked_items: List[Tuple[int, float]], 
                       feature_data: pd.DataFrame,
                       ctr_predictions: np.ndarray, 
                       cvr_predictions: np.ndarray) -> List[Dict]:
        """Format ranking results"""
        
        results = []
        
        for rank, (idx, score) in enumerate(ranked_items):
            item_data = feature_data.iloc[idx]
            
            result = {
                'rank': rank + 1,
                'item_id': item_data.get('item_id', f'item_{idx}'),
                'final_score': float(score),
                'ctr_prediction': float(ctr_predictions[idx]),
                'cvr_prediction': float(cvr_predictions[idx]),
                'expected_revenue': float(cvr_predictions[idx] * item_data.get('item_price', 0)),
                'item_features': {
                    'price': item_data.get('item_price', 0),
                    'rating': item_data.get('item_rating', 0),
                    'category': item_data.get('item_category', 'unknown'),
                    'review_count': item_data.get('item_review_count', 0)
                }
            }
            
            results.append(result)
        
        return results

class ABTestingFramework:
    """A/B testing framework for ranking experiments"""
    
    def __init__(self):
        self.experiments = {}
        
    def create_experiment(self, experiment_id: str, 
                         control_config: RankingConfig,
                         treatment_config: RankingConfig,
                         traffic_split: float = 0.5):
        """Create a new A/B test experiment"""
        
        self.experiments[experiment_id] = {
            'control_config': control_config,
            'treatment_config': treatment_config,
            'traffic_split': traffic_split,
            'results': {'control': [], 'treatment': []}
        }
        
        print(f"Created experiment {experiment_id} with {traffic_split:.1%} traffic to treatment")
    
    def get_user_variant(self, user_id: str, experiment_id: str) -> str:
        """Determine which variant a user should see"""
        
        if experiment_id not in self.experiments:
            return 'control'
        
        # Simple hash-based assignment for consistent user experience
        user_hash = hash(f"{user_id}_{experiment_id}") % 100
        traffic_split = self.experiments[experiment_id]['traffic_split']
        
        return 'treatment' if user_hash < (traffic_split * 100) else 'control'
    
    def log_interaction(self, experiment_id: str, variant: str, 
                       user_id: str, item_id: str, action: str, value: float = 0):
        """Log user interaction for experiment analysis"""
        
        if experiment_id not in self.experiments:
            return
        
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'action': action,  # 'view', 'click', 'conversion'
            'value': value,
            'timestamp': pd.Timestamp.now()
        }
        
        self.experiments[experiment_id]['results'][variant].append(interaction)
    
    def analyze_experiment(self, experiment_id: str) -> Dict:
        """Analyze experiment results"""
        
        if experiment_id not in self.experiments:
            return {}
        
        control_data = pd.DataFrame(self.experiments[experiment_id]['results']['control'])
        treatment_data = pd.DataFrame(self.experiments[experiment_id]['results']['treatment'])
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            return {'error': 'Insufficient data for analysis'}
        
        # Calculate key metrics
        control_metrics = self._calculate_metrics(control_data)
        treatment_metrics = self._calculate_metrics(treatment_data)
        
        # Calculate statistical significance (simplified)
        results = {
            'experiment_id': experiment_id,
            'control_metrics': control_metrics,
            'treatment_metrics': treatment_metrics,
            'lift': {
                'ctr': (treatment_metrics['ctr'] - control_metrics['ctr']) / control_metrics['ctr'] * 100,
                'cvr': (treatment_metrics['cvr'] - control_metrics['cvr']) / control_metrics['cvr'] * 100,
                'revenue_per_user': (treatment_metrics['revenue_per_user'] - control_metrics['revenue_per_user']) / control_metrics['revenue_per_user'] * 100
            }
        }
        
        return results
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate metrics for experiment variant"""
        
        total_users = data['user_id'].nunique()
        total_views = len(data[data['action'] == 'view'])
        total_clicks = len(data[data['action'] == 'click'])
        total_conversions = len(data[data['action'] == 'conversion'])
        total_revenue = data[data['action'] == 'conversion']['value'].sum()
        
        return {
            'users': total_users,
            'views': total_views,
            'clicks': total_clicks,
            'conversions': total_conversions,
            'ctr': total_clicks / total_views if total_views > 0 else 0,
            'cvr': total_conversions / total_clicks if total_clicks > 0 else 0,
            'revenue': total_revenue,
            'revenue_per_user': total_revenue / total_users if total_users > 0 else 0
        }

def demo_ranking_system():
    """Demonstrate the ranking system"""
    print("="*60)
    print("RANKING SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create ranking configuration
    config = RankingConfig(
        ctr_weight=0.4,
        cvr_weight=0.6,
        revenue_weight=1.0,
        top_k=20
    )
    
    # Initialize ranking system
    ranking_system = RankingSystem(config)
    
    # Load models (in practice, you'd load actual trained models)
    model_paths = {
        'ctr_xgb': 'ctr_xgboost_model.model',
        'cvr_xgb': 'cvr_xgboost_model.model'
    }
    # ranking_system.load_models(model_paths)  # Commented out for demo
    
    # Sample user features
    user_features = {
        'age': 32,
        'gender': 'F',
        'location': 'US',
        'device_type': 'mobile',
        'previous_purchases': 5,
        'lifetime_value': 250.0,
        'recency_days': 7
    }
    
    # Sample candidate items
    candidate_items = []
    categories = ['electronics', 'fashion', 'home', 'books', 'beauty']
    
    for i in range(100):
        item = {
            'id': f'item_{i}',
            'category': np.random.choice(categories),
            'price': np.random.lognormal(3, 1),
            'rating': np.random.normal(4.2, 0.8).clip(1, 5),
            'review_count': int(np.random.exponential(50))
        }
        candidate_items.append(item)
    
    # Context features
    context_features = {
        'hour': 14,
        'day_of_week': 2,
        'is_weekend': 0,
        'season': 'summer'
    }
    
    # Rank items
    ranked_items = ranking_system.rank_items(user_features, candidate_items, context_features)
    
    # Display results
    print(f"\nTop 10 Ranked Items:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Item ID':<10} {'Score':<8} {'CTR':<8} {'CVR':<8} {'Revenue':<10} {'Category':<12}")
    print("-" * 80)
    
    for item in ranked_items[:10]:
        print(f"{item['rank']:<4} {item['item_id']:<10} {item['final_score']:.4f}   "
              f"{item['ctr_prediction']:.4f}   {item['cvr_prediction']:.4f}   "
              f"${item['expected_revenue']:.2f}     {item['item_features']['category']:<12}")
    
    # Demonstrate A/B testing
    print(f"\n{'='*60}")
    print("A/B TESTING DEMONSTRATION")
    print("="*60)
    
    ab_framework = ABTestingFramework()
    
    # Create experiment
    control_config = RankingConfig(ctr_weight=0.5, cvr_weight=0.5)
    treatment_config = RankingConfig(ctr_weight=0.3, cvr_weight=0.7)  # More focus on CVR
    
    ab_framework.create_experiment('cvr_focus_test', control_config, treatment_config, 0.5)
    
    # Simulate some interactions
    for user_id in range(100):
        variant = ab_framework.get_user_variant(f'user_{user_id}', 'cvr_focus_test')
        
        # Simulate interactions
        ab_framework.log_interaction('cvr_focus_test', variant, f'user_{user_id}', 'item_1', 'view')
        
        if np.random.random() < 0.15:  # 15% CTR
            ab_framework.log_interaction('cvr_focus_test', variant, f'user_{user_id}', 'item_1', 'click')
            
            if np.random.random() < 0.08:  # 8% CVR
                revenue = np.random.lognormal(3, 0.5)
                ab_framework.log_interaction('cvr_focus_test', variant, f'user_{user_id}', 'item_1', 'conversion', revenue)
    
    # Analyze results
    results = ab_framework.analyze_experiment('cvr_focus_test')
    
    if 'error' not in results:
        print(f"Experiment Results:")
        print(f"Control CTR: {results['control_metrics']['ctr']:.4f}")
        print(f"Treatment CTR: {results['treatment_metrics']['ctr']:.4f}")
        print(f"CTR Lift: {results['lift']['ctr']:.2f}%")
        print(f"Control CVR: {results['control_metrics']['cvr']:.4f}")
        print(f"Treatment CVR: {results['treatment_metrics']['cvr']:.4f}")
        print(f"CVR Lift: {results['lift']['cvr']:.2f}%")
        print(f"Revenue per User Lift: {results['lift']['revenue_per_user']:.2f}%")
    
    return ranked_items, results

if __name__ == "__main__":
    ranked_items, ab_results = demo_ranking_system()
