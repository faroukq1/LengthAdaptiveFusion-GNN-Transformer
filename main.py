import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import random
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class MovieLens1MDataset(Dataset):
    """MovieLens 1M Dataset"""
    def __init__(self, sequences, max_len=200, num_items=None):
        self.sequences = sequences
        self.max_len = max_len
        self.num_items = num_items
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        
        # Input sequence (all except last item)
        input_seq = seq[:-1]
        # Target item (last item)
        target = seq[-1]
        
        # Pad sequence if shorter than max_len
        if len(input_seq) < self.max_len:
            pad_len = self.max_len - len(input_seq)
            input_seq = [0] * pad_len + input_seq
        
        return torch.LongTensor(input_seq), torch.LongTensor([target])

class AttentionLayer(nn.Module):
    """Self-Attention Layer for SES4Rec"""
    def __init__(self, hidden_size, num_heads=1, dropout_rate=0.5):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Context vector
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + context)
        
        return output, attention_weights

class SES4Rec(nn.Module):
    """SES4Rec: Self-Attentive Sequential Recommendation Model"""
    def __init__(self, num_items, hidden_size=50, num_blocks=2, num_heads=1, 
                 dropout_rate=0.5, max_len=200):
        super(SES4Rec, self).__init__()
        
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        # Item embedding layer
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        
        # Positional encoding
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        
        # Attention blocks
        self.attention_layers = nn.ModuleList([
            AttentionLayer(hidden_size, num_heads, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Final prediction layer
        self.output_layer = nn.Linear(hidden_size, num_items + 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, input_seqs):
        batch_size, seq_len = input_seqs.size()
        
        # Create attention mask
        attention_mask = (input_seqs > 0).unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.float()
        
        # Get item embeddings
        item_emb = self.item_embeddings(input_seqs)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embeddings(positions)
        
        # Combine embeddings
        seq_emb = item_emb + pos_emb
        seq_emb = self.dropout(seq_emb)
        
        # Apply attention blocks
        for attention_layer in self.attention_layers:
            seq_emb, _ = attention_layer(seq_emb, attention_mask)
        
        # Apply feed-forward network
        seq_emb = self.ffn(seq_emb)
        
        # Get the last hidden state for prediction
        last_hidden = seq_emb[:, -1, :]
        
        # Final prediction
        output = self.output_layer(last_hidden)
        
        return output

class MetricsCalculator:
    """Calculate NDCG and HR metrics"""
    @staticmethod
    def get_metrics(model, dataloader, device, k=10):
        model.eval()
        all_ndcg = []
        all_hr = []
        
        with torch.no_grad():
            for batch_seqs, batch_targets in tqdm(dataloader, desc='Evaluating', leave=False):
                batch_seqs = batch_seqs.to(device)
                batch_targets = batch_targets.to(device).squeeze()
                
                # Get predictions
                predictions = model(batch_seqs)
                
                # Get top-k items
                _, topk_indices = torch.topk(predictions, k, dim=1)
                
                # Calculate HR and NDCG
                for i in range(len(batch_targets)):
                    target = batch_targets[i].item()
                    topk = topk_indices[i].cpu().numpy()
                    
                    # Hit Rate
                    hr = 1.0 if target in topk else 0.0
                    all_hr.append(hr)
                    
                    # NDCG
                    if target in topk:
                        rank = np.where(topk == target)[0][0] + 1
                        ndcg = 1.0 / np.log2(rank + 1)
                    else:
                        ndcg = 0.0
                    all_ndcg.append(ndcg)
        
        avg_ndcg = np.mean(all_ndcg)
        avg_hr = np.mean(all_hr)
        
        return avg_ndcg, avg_hr

def load_movielens_1m(data_path='ml-1m'):
    """Load and preprocess MovieLens 1M dataset"""
    # Load ratings
    ratings = pd.read_csv(
        os.path.join(data_path, 'ratings.dat'),
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    
    # Filter ratings >= 4 (as done in many papers)
    ratings = ratings[ratings['rating'] >= 4]
    
    # Sort by timestamp
    ratings = ratings.sort_values(['user_id', 'timestamp'])
    
    # Create user sequences
    user_sequences = []
    for user_id, group in ratings.groupby('user_id'):
        # Convert movie IDs to 1-indexed (0 is for padding)
        seq = (group['movie_id'].values + 1).tolist()
        if len(seq) >= 3:  # Only keep users with at least 3 interactions
            user_sequences.append(seq)
    
    # Create train/valid/test splits (last item for test, second last for valid)
    train_seqs = []
    valid_seqs = []
    test_seqs = []
    
    for seq in user_sequences:
        if len(seq) >= 3:
            train_seqs.append(seq[:-2])
            valid_seqs.append(seq[:-1])
            test_seqs.append(seq)
        else:
            train_seqs.append(seq[:-1])
            valid_seqs.append(seq)
            test_seqs.append(seq)
    
    # Get number of unique items
    all_items = ratings['movie_id'].unique()
    num_items = len(all_items)
    
    return train_seqs, valid_seqs, test_seqs, num_items

def train_model(model, train_loader, valid_loader, test_loader, device, 
                num_epochs=1000, lr=0.001, l2_reg=0.0):
    """Train the SES4Rec model"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    metrics_calculator = MetricsCalculator()
    best_valid_ndcg = 0
    best_test_metrics = None
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # Training phase
        for batch_seqs, batch_targets in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            batch_seqs = batch_seqs.to(device)
            batch_targets = batch_targets.to(device).squeeze()
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_seqs)
            loss = criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation phase
        valid_ndcg, valid_hr = metrics_calculator.get_metrics(model, valid_loader, device, k=10)
        
        # Test phase (only when validation improves)
        if valid_ndcg > best_valid_ndcg:
            best_valid_ndcg = valid_ndcg
            test_ndcg, test_hr = metrics_calculator.get_metrics(model, test_loader, device, k=10)
            best_test_metrics = (test_ndcg, test_hr)
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_ndcg': valid_ndcg,
                'valid_hr': valid_hr,
                'test_ndcg': test_ndcg,
                'test_hr': test_hr,
            }, 'best_ses4rec_model.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"epoch:{epoch}, time: {epoch_time:.6f}(s), "
              f"valid (NDCG@10: {valid_ndcg:.4f}, HR@10: {valid_hr:.4f}), "
              f"test (NDCG@10: {test_ndcg:.4f}, HR@10: {test_hr:.4f})")
        
        # Save metrics to file
        with open('training_metrics.txt', 'a') as f:
            f.write(f"epoch:{epoch}, time: {epoch_time:.6f}(s), "
                    f"valid (NDCG@10: {valid_ndcg:.4f}, HR@10: {valid_hr:.4f}), "
                    f"test (NDCG@10: {test_ndcg:.4f}, HR@10: {test_hr:.4f})\n")
    
    return best_test_metrics

def main():
    """Main training function"""
    print("Loading MovieLens 1M dataset...")
    train_seqs, valid_seqs, test_seqs, num_items = load_movielens_1m('ml-1m')
    
    print(f"Number of users: {len(train_seqs)}")
    print(f"Number of items: {num_items}")
    print(f"Average sequence length: {np.mean([len(seq) for seq in train_seqs]):.2f}")
    
    # Create datasets
    max_len = 200
    train_dataset = MovieLens1MDataset(train_seqs, max_len, num_items)
    valid_dataset = MovieLens1MDataset(valid_seqs, max_len, num_items)
    test_dataset = MovieLens1MDataset(test_seqs, max_len, num_items)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model hyperparameters (from SES4Rec paper)
    hidden_size = 50
    num_blocks = 2
    num_heads = 1
    dropout_rate = 0.5
    l2_reg = 0.0
    lr = 0.001
    num_epochs = 1000
    
    # Create model
    model = SES4Rec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        max_len=max_len
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Clear previous metrics file
    with open('training_metrics.txt', 'w') as f:
        f.write("Training Metrics Log\n")
        f.write("="*50 + "\n")
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    best_test_metrics = train_model(
        model, train_loader, valid_loader, test_loader, device,
        num_epochs=num_epochs, lr=lr, l2_reg=l2_reg
    )
    
    total_time = time.time() - start_time
    
    # Load best model
    checkpoint = torch.load('best_ses4rec_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    metrics_calculator = MetricsCalculator()
    final_test_ndcg, final_test_hr = metrics_calculator.get_metrics(model, test_loader, device, k=10)
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Total training time: {total_time:.6f} seconds")
    print(f"Test NDCG@10: {final_test_ndcg:.4f}")
    print(f"Test HR@10: {final_test_hr:.4f}")
    print("="*50)
    
    # Save final metrics
    results = {
        'test_ndcg@10': final_test_ndcg,
        'test_hr@10': final_test_hr,
        'total_time': total_time,
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'hyperparameters': {
            'hidden_size': hidden_size,
            'num_blocks': num_blocks,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'max_len': max_len,
            'batch_size': batch_size,
            'learning_rate': lr,
            'l2_reg': l2_reg,
        }
    }
    
    with open('final_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save metrics in the requested format
    with open('final_metrics.txt', 'w') as f:
        f.write(f"Evaluating........................................................................................................................")
        f.write(f"epoch:{checkpoint['epoch']}, time: {total_time:.6f}(s), ")
        f.write(f"valid (NDCG@10: {checkpoint['valid_ndcg']:.4f}, HR@10: {checkpoint['valid_hr']:.4f}), ")
        f.write(f"test (NDCG@10: {final_test_ndcg:.4f}, HR@10: {final_test_hr:.4f})\n")
        f.write("Done\n")
    
    print("\nResults saved to:")
    print("- final_results.pkl (detailed results)")
    print("- final_metrics.txt (formatted metrics)")
    print("- best_ses4rec_model.pth (trained model)")
    print("- training_metrics.txt (all epoch metrics)")

def download_movielens_1m():
    """Download MovieLens 1M dataset if not exists"""
    import requests
    import zipfile
    import io
    
    if not os.path.exists('ml-1m'):
        print("Downloading MovieLens 1M dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        response = requests.get(url)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall('.')
        
        print("Dataset downloaded and extracted to 'ml-1m' folder")

if __name__ == "__main__":
    # Download dataset if needed
    if not os.path.exists('ml-1m'):
        download_movielens_1m()
    
    # Run main training
    main()