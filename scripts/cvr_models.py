import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')

class CVRXGBoostModel:
    def __init__(self, params=None):
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,  # Slightly shallower for CVR
            'learning_rate': 0.05,  # Lower learning rate for CVR
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,  # Higher min_child_weight for CVR
            'reg_alpha': 0.2,  # More regularization
            'reg_lambda': 2,
            'scale_pos_weight': 10,  # Handle class imbalance in CVR
            'random_state': 42,
            'n_jobs': -1
        }
        self.params = params if params else self.default_params
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val, y_val, num_rounds=1500, early_stopping_rounds=100):
        """Train XGBoost model for CVR prediction"""
        print("Training XGBoost CVR model...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Positive rate: {y_train.mean():.4f}")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up evaluation list
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evallist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='weight')
        
        print("XGBoost CVR model training completed!")
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        auc_score = roc_auc_score(y_test, predictions)
        logloss = log_loss(y_test, predictions)
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'auc': auc_score,
            'logloss': logloss,
            'pr_auc': pr_auc
        }
        
        print(f"XGBoost CVR Model Performance:")
        print(f"AUC: {auc_score:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save_model(filepath)
        
        # Save feature importance
        importance_df = pd.DataFrame(
            list(self.feature_importance.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        importance_df.to_csv(filepath.replace('.model', '_feature_importance.csv'), index=False)
        print(f"XGBoost CVR model saved to {filepath}")

class CVRNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.4):
        super(CVRNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate if i < len(hidden_dims) - 1 else dropout_rate * 0.5)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        return self.network(x).squeeze()

class CVRNeuralNetworkTrainer:
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.4, learning_rate=0.0005):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = CVRNeuralNetwork(input_dim, hidden_dims, dropout_rate).to(self.device)
        
        # Use weighted loss for imbalanced CVR data
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=8, factor=0.5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=512, early_stopping_patience=15):
        """Train neural network for CVR prediction"""
        print("Training Neural Network CVR model...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Positive rate: {y_train.mean():.4f}")
        
        # Calculate class weights for imbalanced data
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        print(f"Positive weight: {pos_weight:.2f}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
        
        # Create weighted sampler for imbalanced data
        class_counts = np.bincount(y_train.values.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.values.astype(int)]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Apply class weighting
                weights = torch.where(batch_y == 1, pos_weight, 1.0)
                loss = nn.functional.binary_cross_entropy(outputs, batch_y, weight=weights)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                self.val_losses.append(val_loss)
                
                # Calculate AUC
                val_predictions = val_outputs.cpu().numpy()
                val_auc = roc_auc_score(y_val.values, val_predictions)
                self.val_aucs.append(val_auc)
            
            # Learning rate scheduling
            self.scheduler.step(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_cvr_nn_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cvr_nn_model.pth'))
        print(f"Neural Network CVR model training completed! Best Val AUC: {best_val_auc:.4f}")
        
        return best_val_auc
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        auc_score = roc_auc_score(y_test, predictions)
        logloss = log_loss(y_test, predictions)
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'auc': auc_score,
            'logloss': logloss,
            'pr_auc': pr_auc
        }
        
        print(f"Neural Network CVR Model Performance:")
        print(f"AUC: {auc_score:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('CVR Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # AUC plot
        ax2.plot(self.val_aucs, label='Validation AUC', color='green')
        ax2.set_title('CVR Validation AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('cvr_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def load_and_prepare_cvr_data(sample_size=500000):
    """Load and prepare data for CVR model training (only clicked samples)"""
    print(f"Loading and preparing CVR data...")
    
    # For demonstration, we'll create sample data with realistic CVR patterns
    # In practice, this would load from the generated dataset chunks and filter for clicks
    np.random.seed(42)
    
    # Generate sample data (this would be replaced with actual data loading)
    # Only include samples where click = 1
    data = pd.DataFrame({
        'user_id': np.random.randint(1, 100000, sample_size),
        'item_id': np.random.randint(1, 10000, sample_size),
        'user_age': np.random.normal(35, 12, sample_size).clip(18, 80).astype(int),
        'user_gender': np.random.choice(['M', 'F'], sample_size),
        'user_location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], sample_size),
        'user_device_type': np.random.choice(['mobile', 'desktop', 'tablet'], sample_size),
        'item_category': np.random.choice(['electronics', 'fashion', 'home', 'books'], sample_size),
        'item_price': np.random.lognormal(3, 1, sample_size),
        'item_rating': np.random.normal(4.2, 0.8, sample_size).clip(1, 5),
        'item_review_count': np.random.exponential(50, sample_size).astype(int),
        'hour': np.random.randint(0, 24, sample_size),
        'day_of_week': np.random.randint(0, 7, sample_size),
        'is_weekend': np.random.choice([0, 1], sample_size),
        'season': np.random.choice(['spring', 'summer', 'fall', 'winter'], sample_size),
        'user_session_length': np.random.exponential(15, sample_size),
        'user_page_views': np.random.poisson(8, sample_size),
        'user_previous_purchases': np.random.poisson(2, sample_size),
        'user_cart_size': np.random.poisson(1.5, sample_size),
        'user_lifetime_value': np.random.lognormal(5, 1, sample_size),
        'user_recency_days': np.random.exponential(30, sample_size),
        'user_frequency_score': np.random.gamma(2, 2, sample_size),
        'click': np.ones(sample_size)  # All samples are clicked
    })
    
    # Generate realistic CVR labels (more challenging than CTR)
    cvr_logits = (
        -2.2 +  # Lower base conversion rate
        0.4 * (data['item_rating'] - 3) +
        -0.4 * np.log(data['item_price']) +  # Strong price sensitivity
        0.3 * (data['user_previous_purchases'] > 0) +
        0.2 * (data['user_cart_size'] > 0) +
        0.15 * (data['user_lifetime_value'] > data['user_lifetime_value'].quantile(0.7)) +
        -0.1 * data['user_recency_days'] / 30 +
        0.2 * data['user_frequency_score'] +
        np.random.normal(0, 0.25, sample_size)
    )
    
    data['conversion'] = np.random.binomial(1, 1 / (1 + np.exp(-cvr_logits)))
    
    print(f"CVR data loaded. Conversion rate: {data['conversion'].mean():.4f}")
    return data

def train_cvr_models():
    """Main function to train both XGBoost and Neural Network CVR models"""
    print("="*60)
    print("TRAINING CVR PREDICTION MODELS")
    print("="*60)
    
    # Load and prepare data (only clicked samples)
    data = load_and_prepare_cvr_data(500000)  # 500K clicked samples for demo
    
    # Feature engineering
    fe = FeatureEngineer()
    engineered_data = fe.engineer_features(data)
    
    # Get feature matrix for CVR (only clicked samples)
    X, y = fe.get_feature_matrix(engineered_data, 'cvr')
    
    print(f"CVR Feature matrix shape: {X.shape}")
    print(f"CVR Target distribution: {y.value_counts(normalize=True)}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"CVR Train set: {X_train.shape}")
    print(f"CVR Validation set: {X_val.shape}")
    print(f"CVR Test set: {X_test.shape}")
    
    # Train XGBoost model
    print("\n" + "="*40)
    print("TRAINING XGBOOST CVR MODEL")
    print("="*40)
    
    xgb_model = CVRXGBoostModel()
    xgb_model.train(X_train, y_train, X_val, y_val)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save_model('cvr_xgboost_model.model')
    
    # Train Neural Network model
    print("\n" + "="*40)
    print("TRAINING NEURAL NETWORK CVR MODEL")
    print("="*40)
    
    nn_trainer = CVRNeuralNetworkTrainer(
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64, 32],
        dropout_rate=0.4,
        learning_rate=0.0005
    )
    
    best_auc = nn_trainer.train(X_train, y_train, X_val, y_val, epochs=150, batch_size=512)
    nn_metrics = nn_trainer.evaluate(X_test, y_test)
    
    # Plot training history
    nn_trainer.plot_training_history()
    
    # Save models and results
    results = {
        'xgboost_metrics': xgb_metrics,
        'neural_network_metrics': nn_metrics,
        'feature_count': X_train.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'conversion_rate': float(y.mean()),
        'best_cvr_auc': max(xgb_metrics['auc'], nn_metrics['auc'])
    }
    
    with open('cvr_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature engineering config
    fe.save_feature_config('cvr_feature_config.json')
    
    print("\n" + "="*60)
    print("CVR MODEL TRAINING RESULTS")
    print("="*60)
    print(f"XGBoost CVR AUC: {xgb_metrics['auc']:.4f}")
    print(f"Neural Network CVR AUC: {nn_metrics['auc']:.4f}")
    print(f"Best CVR AUC: {results['best_cvr_auc']:.4f}")
    print(f"Best CVR model: {'XGBoost' if xgb_metrics['auc'] > nn_metrics['auc'] else 'Neural Network'}")
    print(f"Conversion rate: {results['conversion_rate']:.4f}")
    
    return results

if __name__ == "__main__":
    results = train_cvr_models()
