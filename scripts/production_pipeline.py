import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from ranking_system import RankingSystem, RankingConfig, ABTestingFramework
import redis
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionRequest:
    """Request format for ranking API"""
    user_id: str
    user_features: Dict
    candidate_items: List[Dict]
    context_features: Optional[Dict] = None
    experiment_id: Optional[str] = None
    top_k: int = 50

@dataclass
class PredictionResponse:
    """Response format for ranking API"""
    user_id: str
    ranked_items: List[Dict]
    experiment_variant: Optional[str] = None
    model_version: str = "v1.0"
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class ModelCache:
    """Redis-based caching for model predictions"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=3600):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.ttl = ttl  # Time to live in seconds
        
    def get_prediction(self, cache_key: str) -> Optional[Dict]:
        """Get cached prediction"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    def set_prediction(self, cache_key: str, prediction: Dict):
        """Cache prediction"""
        try:
            serialized_data = pickle.dumps(prediction)
            self.redis_client.setex(cache_key, self.ttl, serialized_data)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def generate_cache_key(self, user_id: str, item_ids: List[str], context_hash: str) -> str:
        """Generate cache key for prediction"""
        items_hash = hash(tuple(sorted(item_ids)))
        return f"prediction:{user_id}:{items_hash}:{context_hash}"

class FeatureStore:
    """Feature store for real-time feature serving"""
    
    def __init__(self):
        self.user_features_cache = {}
        self.item_features_cache = {}
        
    async def get_user_features(self, user_id: str) -> Dict:
        """Get user features from feature store"""
        
        # In production, this would query a real feature store
        # For demo, we'll simulate feature retrieval
        if user_id in self.user_features_cache:
            return self.user_features_cache[user_id]
        
        # Simulate async feature retrieval
        await asyncio.sleep(0.01)  # Simulate network latency
        
        # Generate realistic user features
        features = {
            'age': np.random.randint(18, 70),
            'gender': np.random.choice(['M', 'F']),
            'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE']),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
            'previous_purchases': np.random.poisson(3),
            'lifetime_value': np.random.lognormal(5, 1),
            'recency_days': np.random.exponential(30),
            'frequency_score': np.random.gamma(2, 2)
        }
        
        self.user_features_cache[user_id] = features
        return features
    
    async def get_item_features(self, item_ids: List[str]) -> List[Dict]:
        """Get item features from feature store"""
        
        features_list = []
        
        for item_id in item_ids:
            if item_id in self.item_features_cache:
                features_list.append(self.item_features_cache[item_id])
                continue
            
            # Simulate async feature retrieval
            await asyncio.sleep(0.001)  # Simulate network latency
            
            # Generate realistic item features
            features = {
                'id': item_id,
                'category': np.random.choice(['electronics', 'fashion', 'home', 'books', 'beauty']),
                'price': np.random.lognormal(3, 1),
                'rating': np.random.normal(4.2, 0.8).clip(1, 5),
                'review_count': int(np.random.exponential(50)),
                'brand': f'brand_{np.random.randint(1, 100)}',
                'availability': np.random.choice([True, False], p=[0.9, 0.1])
            }
            
            self.item_features_cache[item_id] = features
            features_list.append(features)
        
        return features_list

class ModelMonitoring:
    """Model performance monitoring and alerting"""
    
    def __init__(self):
        self.metrics_buffer = []
        self.alert_thresholds = {
            'avg_ctr_prediction': (0.05, 0.3),  # (min, max)
            'avg_cvr_prediction': (0.01, 0.2),
            'prediction_latency': (0, 100),  # milliseconds
            'error_rate': (0, 0.05)  # 5% max error rate
        }
        
    def log_prediction_metrics(self, metrics: Dict):
        """Log prediction metrics for monitoring"""
        metrics['timestamp'] = datetime.now()
        self.metrics_buffer.append(metrics)
        
        # Keep only recent metrics (last 1000 predictions)
        if len(self.metrics_buffer) > 1000:
            self.metrics_buffer = self.metrics_buffer[-1000:]
        
        # Check for alerts
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: Dict):
        """Check if metrics exceed alert thresholds"""
        
        for metric_name, (min_val, max_val) in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if value < min_val or value > max_val:
                    logger.warning(f"ALERT: {metric_name} = {value} is outside threshold [{min_val}, {max_val}]")
    
    def get_recent_metrics(self, minutes: int = 60) -> Dict:
        """Get aggregated metrics for recent time period"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_buffer if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {
            'total_predictions': len(recent_metrics),
            'avg_ctr_prediction': np.mean([m.get('avg_ctr_prediction', 0) for m in recent_metrics]),
            'avg_cvr_prediction': np.mean([m.get('avg_cvr_prediction', 0) for m in recent_metrics]),
            'avg_latency_ms': np.mean([m.get('prediction_latency', 0) for m in recent_metrics]),
            'error_rate': np.mean([m.get('error_occurred', 0) for m in recent_metrics]),
            'time_period_minutes': minutes
        }
        
        return aggregated

class ProductionRankingService:
    """Production-ready ranking service"""
    
    def __init__(self, config_path: str = 'ranking_config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        self.config = RankingConfig(**config_dict)
        
        # Initialize components
        self.ranking_system = RankingSystem(self.config)
        self.feature_store = FeatureStore()
        self.model_cache = ModelCache()
        self.ab_framework = ABTestingFramework()
        self.monitoring = ModelMonitoring()
        
        # Load models
        self._load_models()
        
        logger.info("Production ranking service initialized")
    
    def _load_models(self):
        """Load trained models"""
        try:
            model_paths = {
                'ctr_xgb': 'models/ctr_xgboost_model.model',
                'cvr_xgb': 'models/cvr_xgboost_model.model'
            }
            # self.ranking_system.load_models(model_paths)  # Commented for demo
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Main prediction endpoint"""
        start_time = datetime.now()
        
        try:
            # Get user features
            if not request.user_features:
                request.user_features = await self.feature_store.get_user_features(request.user_id)
            
            # Get item features
            item_ids = [item.get('id', f'item_{i}') for i, item in enumerate(request.candidate_items)]
            item_features = await self.feature_store.get_item_features(item_ids)
            
            # Update candidate items with features
            for i, features in enumerate(item_features):
                request.candidate_items[i].update(features)
            
            # Check cache
            context_hash = hash(str(request.context_features)) if request.context_features else "no_context"
            cache_key = self.model_cache.generate_cache_key(request.user_id, item_ids, context_hash)
            
            cached_result = self.model_cache.get_prediction(cache_key)
            if cached_result:
                logger.info(f"Cache hit for user {request.user_id}")
                return PredictionResponse(**cached_result)
            
            # Determine experiment variant
            experiment_variant = None
            if request.experiment_id:
                experiment_variant = self.ab_framework.get_user_variant(
                    request.user_id, request.experiment_id
                )
            
            # Rank items
            ranked_items = self.ranking_system.rank_items(
                request.user_features,
                request.candidate_items,
                request.context_features
            )
            
            # Create response
            response = PredictionResponse(
                user_id=request.user_id,
                ranked_items=ranked_items[:request.top_k],
                experiment_variant=experiment_variant
            )
            
            # Cache result
            self.model_cache.set_prediction(cache_key, asdict(response))
            
            # Log metrics
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000
            metrics = {
                'user_id': request.user_id,
                'num_items': len(request.candidate_items),
                'prediction_latency': prediction_time,
                'avg_ctr_prediction': np.mean([item['ctr_prediction'] for item in ranked_items]),
                'avg_cvr_prediction': np.mean([item['cvr_prediction'] for item in ranked_items]),
                'error_occurred': 0
            }
            self.monitoring.log_prediction_metrics(metrics)
            
            logger.info(f"Prediction completed for user {request.user_id} in {prediction_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Prediction error for user {request.user_id}: {e}")
            
            # Log error metrics
            error_metrics = {
                'user_id': request.user_id,
                'error_occurred': 1,
                'prediction_latency': (datetime.now() - start_time).total_seconds() * 1000
            }
            self.monitoring.log_prediction_metrics(error_metrics)
            
            # Return fallback response
            return self._fallback_response(request)
    
    def _fallback_response(self, request: PredictionRequest) -> PredictionResponse:
        """Fallback response when prediction fails"""
        
        # Simple fallback: random ranking with basic scores
        fallback_items = []
        for i, item in enumerate(request.candidate_items[:request.top_k]):
            fallback_item = {
                'rank': i + 1,
                'item_id': item.get('id', f'item_{i}'),
                'final_score': np.random.uniform(0.1, 0.9),
                'ctr_prediction': np.random.uniform(0.05, 0.25),
                'cvr_prediction': np.random.uniform(0.01, 0.15),
                'expected_revenue': np.random.uniform(1, 50),
                'item_features': item
            }
            fallback_items.append(fallback_item)
        
        return PredictionResponse(
            user_id=request.user_id,
            ranked_items=fallback_items,
            model_version="fallback"
        )
    
    async def log_interaction(self, user_id: str, item_id: str, action: str, 
                            experiment_id: str = None, value: float = 0):
        """Log user interaction for model feedback"""
        
        if experiment_id:
            variant = self.ab_framework.get_user_variant(user_id, experiment_id)
            self.ab_framework.log_interaction(experiment_id, variant, user_id, item_id, action, value)
        
        # Log to monitoring system
        interaction_metrics = {
            'user_id': user_id,
            'item_id': item_id,
            'action': action,
            'value': value
        }
        logger.info(f"Logged interaction: {interaction_metrics}")
    
    def get_health_status(self) -> Dict:
        """Get service health status"""
        
        recent_metrics = self.monitoring.get_recent_metrics(60)  # Last 60 minutes
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'recent_metrics': recent_metrics,
            'model_version': 'v1.0',
            'uptime_hours': 24  # Placeholder
        }
        
        # Check if service is healthy
        if recent_metrics.get('error_rate', 0) > 0.1:  # 10% error rate threshold
            health_status['status'] = 'unhealthy'
            health_status['issues'] = ['High error rate']
        
        return health_status

async def demo_production_service():
    """Demonstrate production ranking service"""
    print("="*60)
    print("PRODUCTION RANKING SERVICE DEMO")
    print("="*60)
    
    # Create service configuration
    config = {
        'ctr_weight': 0.4,
        'cvr_weight': 0.6,
        'revenue_weight': 1.0,
        'diversity_weight': 0.1,
        'freshness_weight': 0.05,
        'popularity_weight': 0.1,
        'min_ctr_threshold': 0.01,
        'min_cvr_threshold': 0.005,
        'max_price_threshold': 1000.0,
        'top_k': 50,
        'diversification_window': 10
    }
    
    # Save config
    with open('ranking_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize service
    service = ProductionRankingService('ranking_config.json')
    
    # Create sample request
    candidate_items = [{'id': f'item_{i}'} for i in range(100)]
    
    request = PredictionRequest(
        user_id='user_123',
        user_features={},  # Will be fetched from feature store
        candidate_items=candidate_items,
        context_features={'hour': 14, 'day_of_week': 2},
        experiment_id='cvr_focus_test',
        top_k=20
    )
    
    # Make prediction
    response = await service.predict(request)
    
    print(f"Prediction for user {response.user_id}:")
    print(f"Model version: {response.model_version}")
    print(f"Experiment variant: {response.experiment_variant}")
    print(f"Top 5 items:")
    
    for item in response.ranked_items[:5]:
        print(f"  Rank {item['rank']}: {item['item_id']} (score: {item['final_score']:.4f})")
    
    # Simulate interactions
    await service.log_interaction('user_123', 'item_0', 'view', 'cvr_focus_test')
    await service.log_interaction('user_123', 'item_0', 'click', 'cvr_focus_test')
    await service.log_interaction('user_123', 'item_0', 'conversion', 'cvr_focus_test', 25.99)
    
    # Check health status
    health = service.get_health_status()
    print(f"\nService Health: {health['status']}")
    print(f"Recent predictions: {health['recent_metrics'].get('total_predictions', 0)}")
    
    return service, response

if __name__ == "__main__":
    # Run demo
    service, response = asyncio.run(demo_production_service())
