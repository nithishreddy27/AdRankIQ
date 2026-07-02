import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import json
from feature_engineering import FeatureEngineer

class MultiTaskCTRCVRNetwork(nn.Module):
    """
    Multi-task learning network for joint CTR and CVR prediction
    Shares lower layers and has separate task-specific heads
    """
    def __init__(self, input_dim, shared_dims=[512, 256], task_dims=[128, 64], dropout_rate=0.3):
        super(MultiTaskCTRCVRNetwork, self).__init__()
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        
        for dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # CTR-specific layers
        ctr_layers = []
        ctr_prev_dim = prev_dim
        
        for dim in task_dims:
            ctr_layers.extend([
                nn.Linear(ctr_prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.8)
            ])
            ctr_prev_dim = dim
        
        ctr_layers.extend([
            nn.Linear(ctr_prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.ctr_head = nn.Sequential(*ctr_layers)
        
        # CVR-specific layers
        cvr_layers = []
        cvr_prev_dim = prev_dim
        
        for dim in task_dims:
            cvr_layers.extend([
                nn.Linear(cvr_prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            cvr_prev_dim = dim
        
        cvr_layers.extend([
            nn.Linear(cvr_prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.cvr_head = nn.Sequential(*cvr_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared_network(x)
        ctr_output = self.ctr_head(shared_features).squeeze()
        cvr_output = self.cvr_head(shared_features).squeeze()
        return ctr_output, cvr_output

class MultiTaskTrainer:
    def __init__(self, input_dim, shared_dims=[512, 256], task_dims=[128, 64], 
                 dropout_rate=0.3, learning_rate=0.001, ctr_weight=1.0, cvr_weight=1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = MultiTaskCTRCVRNetwork(input_dim, shared_dims, task_dims, dropout_rate).to(self.device)
        
        self.ctr_criterion = nn.BCELoss()
        self.cvr_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.7)
        
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight
        
        self.train_losses = []
        self.val_losses = []
        self.ctr_aucs = []
        self.cvr_aucs = []
        
    def train(self, X_train, y_ctr_train, y_cvr_train, X_val, y_ctr_val, y_cvr_val, 
              epochs=100, batch_size=1024, early_stopping_patience=15):
        """Train multi-task model for joint CTR and CVR prediction"""
        print("Training Multi-Task CTR-CVR model...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_ctr_train_tensor = torch.FloatTensor(y_ctr_train.values).to(self.device)
        y_cvr_train_tensor = torch.FloatTensor(y_cvr_train.values).to(self.device)
        
        X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
        y_ctr_val_tensor = torch.FloatTensor(y_ctr_val.values).to(self.device)
        y_cvr_val_tensor = torch.FloatTensor(y_cvr_val.values).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_ctr_train_tensor, y_cvr_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_combined_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y_ctr, batch_y_cvr in train_loader:
                self.optimizer.zero_grad()
                
                ctr_outputs, cvr_outputs = self.model(batch_X)
                
                # Calculate losses
                ctr_loss = self.ctr_criterion(ctr_outputs, batch_y_ctr)
                cvr_loss = self.cvr_criterion(cvr_outputs, batch_y_cvr)
                
                # Combined loss with weights
                total_loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += total_loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                ctr_val_outputs, cvr_val_outputs = self.model(X_val_tensor)
                
                ctr_val_loss = self.ctr_criterion(ctr_val_outputs, y_ctr_val_tensor)
                cvr_val_loss = self.cvr_criterion(cvr_val_outputs, y_cvr_val_tensor)
                total_val_loss = self.ctr_weight * ctr_val_loss + self.cvr_weight * cvr_val_loss
                
                self.val_losses.append(total_val_loss.item())
                
                # Calculate AUCs
                ctr_predictions = ctr_val_outputs.cpu().numpy()
                cvr_predictions = cvr_val_outputs.cpu().numpy()
                
                ctr_auc = roc_auc_score(y_ctr_val.values, ctr_predictions)
                cvr_auc = roc_auc_score(y_cvr_val.values, cvr_predictions)
                
                self.ctr_aucs.append(ctr_auc)
                self.cvr_aucs.append(cvr_auc)
                
                # Combined AUC for early stopping
                combined_auc = (ctr_auc + cvr_auc) / 2
            
            # Learning rate scheduling
            self.scheduler.step(total_val_loss)
            
            # Early stopping
            if combined_auc > best_combined_auc:
                best_combined_auc = combined_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_multitask_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")
                print(f"  CTR AUC: {ctr_auc:.4f}, CVR AUC: {cvr_auc:.4f}, Combined: {combined_auc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_multitask_model.pth'))
        print(f"Multi-task model training completed! Best Combined AUC: {best_combined_auc:.4f}")
        
        return best_combined_auc
    
    def predict(self, X):
        """Make predictions for both CTR and CVR"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            ctr_predictions, cvr_predictions = self.model(X_tensor)
            return ctr_predictions.cpu().numpy(), cvr_predictions.cpu().numpy()
    
    def evaluate(self, X_test, y_ctr_test, y_cvr_test):
        """Evaluate model performance on both tasks"""
        ctr_predictions, cvr_predictions = self.predict(X_test)
        
        ctr_auc = roc_auc_score(y_ctr_test, ctr_predictions)
        cvr_auc = roc_auc_score(y_cvr_test, cvr_predictions)
        
        ctr_logloss = log_loss(y_ctr_test, ctr_predictions)
        cvr_logloss = log_loss(y_cvr_test, cvr_predictions)
        
        metrics = {
            'ctr_auc': ctr_auc,
            'cvr_auc': cvr_auc,
            'ctr_logloss': ctr_logloss,
            'cvr_logloss': cvr_logloss,
            'combined_auc': (ctr_auc + cvr_auc) / 2
        }
        
        print(f"Multi-Task Model Performance:")
        print(f"CTR AUC: {ctr_auc:.4f}, Log Loss: {ctr_logloss:.4f}")
        print(f"CVR AUC: {cvr_auc:.4f}, Log Loss: {cvr_logloss:.4f}")
        print(f"Combined AUC: {metrics['combined_auc']:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history for both tasks"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Multi-Task Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # CTR AUC
        ax2.plot(self.ctr_aucs, label='CTR AUC', color='blue')
        ax2.set_title('CTR Validation AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True)
        
        # CVR AUC
        ax3.plot(self.cvr_aucs, label='CVR AUC', color='red')
        ax3.set_title('CVR Validation AUC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AUC')
        ax3.legend()
        ax3.grid(True)
        
        # Combined AUC
        combined_aucs = [(ctr + cvr) / 2 for ctr, cvr in zip(self.ctr_aucs, self.cvr_aucs)]
        ax4.plot(combined_aucs, label='Combined AUC', color='green')
        ax4.set_title('Combined AUC')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('AUC')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('multitask_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def train_multitask_model():
    """Train multi-task CTR-CVR model"""
    print("="*60)
    print("TRAINING MULTI-TASK CTR-CVR MODEL")
    print("="*60)
    
    # Load and prepare data
    np.random.seed(42)
    sample_size = 800000
    
    # Generate sample data
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
    })
    
    # Generate CTR labels
    ctr_logits = (
        -2.5 + 0.3 * (data['item_rating'] - 3) + 
        0.2 * np.log(data['item_review_count'] + 1) +
        -0.1 * np.log(data['item_price']) +
        np.random.normal(0, 0.3, sample_size)
    )
    data['click'] = np.random.binomial(1, 1 / (1 + np.exp(-ctr_logits)))
    
    # Generate CVR labels (conditional on click)
    cvr_logits = np.where(data['click'] == 1,
        -2.0 + 0.4 * (data['item_rating'] - 3) +
        -0.3 * np.log(data['item_price']) +
        0.3 * (data['user_previous_purchases'] > 0) +
        np.random.normal(0, 0.2, sample_size),
        -10  # Very low probability if no click
    )
    data['conversion'] = np.where(data['click'] == 1, 
                                np.random.binomial(1, 1 / (1 + np.exp(-cvr_logits))), 0)
    
    print(f"CTR: {data['click'].mean():.4f}")
    print(f"CVR (overall): {data['conversion'].mean():.4f}")
    print(f"CVR (given click): {data[data['click']==1]['conversion'].mean():.4f}")
    
    # Feature engineering
    fe = FeatureEngineer()
    engineered_data = fe.engineer_features(data)
    
    # Get feature matrices
    X_ctr, y_ctr = fe.get_feature_matrix(engineered_data, 'ctr')
    X_cvr, y_cvr = fe.get_feature_matrix(engineered_data, 'cvr')
    
    # For multi-task learning, we need the same X for both tasks
    # Use the full dataset and handle CVR masking in the loss
    X = X_ctr
    y_ctr_full = engineered_data['click']
    y_cvr_full = engineered_data['conversion']
    
    print(f"Multi-task feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_temp, y_ctr_train, y_ctr_temp, y_cvr_train, y_cvr_temp = train_test_split(
        X, y_ctr_full, y_cvr_full, test_size=0.3, random_state=42, stratify=y_ctr_full
    )
    X_val, X_test, y_ctr_val, y_ctr_test, y_cvr_val, y_cvr_test = train_test_split(
        X_temp, y_ctr_temp, y_cvr_temp, test_size=0.5, random_state=42, stratify=y_ctr_temp
    )
    
    # Train multi-task model
    trainer = MultiTaskTrainer(
        input_dim=X_train.shape[1],
        shared_dims=[512, 256],
        task_dims=[128, 64],
        dropout_rate=0.3,
        learning_rate=0.001,
        ctr_weight=1.0,
        cvr_weight=2.0  # Higher weight for CVR due to class imbalance
    )
    
    best_auc = trainer.train(X_train, y_ctr_train, y_cvr_train, 
                           X_val, y_ctr_val, y_cvr_val, epochs=100)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_ctr_test, y_cvr_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save results
    results = {
        'multitask_metrics': metrics,
        'feature_count': X_train.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'ctr_rate': float(y_ctr_full.mean()),
        'cvr_rate': float(y_cvr_full.mean())
    }
    
    with open('multitask_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("MULTI-TASK MODEL RESULTS")
    print("="*60)
    print(f"CTR AUC: {metrics['ctr_auc']:.4f}")
    print(f"CVR AUC: {metrics['cvr_auc']:.4f}")
    print(f"Combined AUC: {metrics['combined_auc']:.4f}")
    
    return results

if __name__ == "__main__":
    results = train_multitask_model()
