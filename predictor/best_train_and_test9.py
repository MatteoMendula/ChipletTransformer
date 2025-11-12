#!/usr/bin/env python
# coding: utf-8

"""
sequential_forecaster.py

This script implements a GRU (a type of RNN) to solve the true
forecasting problem: "Given the last 'N' steps of a user's context,
predict the 'nr_ssRsrp' for the *next* step."

This is the best-performing model (RMSE ~12.67).

This script combines all our learnings:
1.  **Model:** Sequential (GRU).
2.  **Task:** Forecasting (predicting T+1 from a history).
3.  **Data Split:** Correct per-user 80/20 timeline split.
4.  **Target:** 'nr_ssRsrp'.
5.  **Handling Zeros:** Replaces 0.0 with a low-value (-140) to
    treat "no connection" as the bottom of the signal range.
6.  *** Autoregressive (ARX) ***
    The model now uses the *history* of 'nr_ssRsrp' as an
    input feature, in addition to the context.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import sys
import warnings
import os # *** NEW: Import os to check for file existence ***

# --- Configuration ---
FILE_PATH = "/home/workstation163/Documents/chetna_paper/Lumos5G-v1.0.csv"
TARGET_VARIABLE = "nr_ssRsrp"

# Features
INPUT_FEATURES_NUM = [
    'latitude',
    'longitude',
    'movingSpeed',
    'compassDirection'
]
INPUT_FEATURES_CAT = [
    'mobility_mode',
    'trajectory_direction',
    'tower_id'
]
ALL_CONTEXT_FEATURES = INPUT_FEATURES_NUM + INPUT_FEATURES_CAT

# Zeros (no signal) will be replaced with this value before scaling
ZERO_REPLACEMENT_VALUE = -140.0

# Model Hyperparameters
SEQUENCE_LENGTH = 10  # How many past steps to look at
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings('ignore', category=UserWarning)

def load_and_preprocess_data(file_path):
    """
    Loads, cleans, preprocesses, and splits the data.
    """
    print(f"Loading and preprocessing data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Could not read file. {e}", file=sys.stderr)
        return None

    # 1. Clean data
    # Handle NaNs in target
    df[TARGET_VARIABLE] = df[TARGET_VARIABLE].fillna(0.0)
    # *** KEY: Replace 0.0 with a low value ***
    df[TARGET_VARIABLE] = df[TARGET_VARIABLE].replace(0.0, ZERO_REPLACEMENT_VALUE)

    # Handle NaNs in numerical inputs
    for col in INPUT_FEATURES_NUM:
        df[col] = df[col].fillna(0.0)

    # Handle NaNs/types in categorical inputs
    for col in INPUT_FEATURES_CAT:
        df[col] = df[col].fillna("MISSING").astype(str)

    # 2. Split data: 80% train, 20% test *for each user*
    print("Splitting each user's session into 80% train / 20% test...")
    train_dfs = []
    test_dfs = []
    
    for run_num in df['run_num'].unique():
        user_df = df[df['run_num'] == run_num].sort_values(by='seq_num')
        if len(user_df) < SEQUENCE_LENGTH + 5: # Need enough data for sequence
            continue
        
        split_idx = int(len(user_df) * 0.8)
        train_dfs.append(user_df.iloc[:split_idx])
        test_dfs.append(user_df.iloc[split_idx:])

    train_df_raw = pd.concat(train_dfs)
    test_df_raw = pd.concat(test_dfs)

    if train_df_raw.empty or test_df_raw.empty:
        print("ERROR: Train or test set is empty.", file=sys.stderr)
        return None

    # 3. Scaling and Encoding
    # Fit scalers/encoders *only* on the training data
    
    # --- Context Features ---
    scaler_features_num = MinMaxScaler()
    encoder_features_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform train
    train_num = scaler_features_num.fit_transform(train_df_raw[INPUT_FEATURES_NUM])
    train_cat = encoder_features_cat.fit_transform(train_df_raw[INPUT_FEATURES_CAT])
    
    # Transform test
    test_num = scaler_features_num.transform(test_df_raw[INPUT_FEATURES_NUM])
    test_cat = encoder_features_cat.transform(test_df_raw[INPUT_FEATURES_CAT])
    
    # --- Target / Autoregressive Feature ---
    scaler_target = MinMaxScaler()
    
    # Fit and transform train target
    train_target_scaled = scaler_target.fit_transform(train_df_raw[[TARGET_VARIABLE]])
    # Transform test target
    test_target_scaled = scaler_target.transform(test_df_raw[[TARGET_VARIABLE]])

    # --- Combine all features ---
    # *** Add scaled target as an input feature ***
    train_features_combined = np.hstack([train_num, train_cat, train_target_scaled])
    test_features_combined = np.hstack([test_num, test_cat, test_target_scaled])

    # Store full data in a dict
    data = {
        'train': {
            'features': train_features_combined,
            'target': train_target_scaled, # Target is just the scaled target
            'run_num': train_df_raw['run_num'].values,
            'seq_num': train_df_raw['seq_num'].values # *** NEW: Pass seq_num
        },
        'test': {
            'features': test_features_combined,
            'target': test_target_scaled,
            'run_num': test_df_raw['run_num'].values,
            'seq_num': test_df_raw['seq_num'].values # *** NEW: Pass seq_num
        },
        'scaler_target': scaler_target,
        'num_features': train_features_combined.shape[1]
    }
    return data

def create_sequences(features, target, run_nums, seq_nums, seq_length):
    """
    Converts flat data into sequences for an RNN.
    This version correctly handles user boundaries.
    X features include the target (for autoregression)
    y target is the *next* step's target
    
    Returns:
        X_seq (np.array), y_seq (np.array), seq_info (pd.DataFrame)
    """
    X_seq, y_seq = [], []
    seq_info = [] # To store (run_num, seq_num)
    
    # Group data by run_num to process users individually
    user_indices = {}
    for i, run_num in enumerate(run_nums):
        if run_num not in user_indices:
            user_indices[run_num] = []
        user_indices[run_num].append(i)

    for run_num, indices in user_indices.items():
        if len(indices) < seq_length + 1:
            continue
            
        user_features = features[indices]
        user_target = target[indices] 
        user_seq_nums = seq_nums[indices] # *** NEW: Get seq_nums for this user
        
        # Create sequences for this user
        for i in range(len(user_features) - seq_length):
            # Input sequence (features + target history)
            X_seq.append(user_features[i : i + seq_length])
            
            # Target (the single *next* target value)
            target_idx = i + seq_length
            y_seq.append(user_target[target_idx])
            
            # *** NEW: Store info for this target row ***
            seq_info.append({
                'run_num': run_num, 
                'seq_num': user_seq_nums[target_idx]
            })

    return np.array(X_seq), np.array(y_seq), pd.DataFrame(seq_info)

class RSRPSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# *** NEW: Using GRUModel ***
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU returns: output, hidden_state
        out, _ = self.gru(x, h0)
        
        # Get the output of the last time step
        out = out[:, -1, :]
        
        # Pass through the final fully connected layer
        out = self.fc(out)
        return out

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    print("Starting training...")
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
            
        scheduler.step(avg_loss) # Step the scheduler on validation loss
        
    print("Training finished.")

def evaluate_model(model, test_loader, scaler_target, device):
    print("Evaluating model...")
    model.eval()
    
    all_preds_scaled = []
    all_targets_scaled = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            outputs = model(sequences)
            
            all_preds_scaled.extend(outputs.cpu().numpy())
            all_targets_scaled.extend(targets.cpu().numpy())
            
    all_preds_scaled = np.array(all_preds_scaled)
    all_targets_scaled = np.array(all_targets_scaled)

    # Inverse transform to get actual RSRP values
    all_preds = scaler_target.inverse_transform(all_preds_scaled)
    all_targets = scaler_target.inverse_transform(all_targets_scaled)

    # Calculate final RMSE
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    print(f"\nTest RMSE on actual nr_ssRsrp values: {rmse:.4f}\n")

    # Show sample predictions
    print("Sample Predictions (Actual vs. Predicted):")
    if len(all_targets) > 10:
        sample_indices = np.random.choice(len(all_targets), 10, replace=False)
        for i in sample_indices:
            # Clamp predictions at the "zero" value
            pred_val = all_preds[i][0]
            if pred_val < ZERO_REPLACEMENT_VALUE:
                pred_val = ZERO_REPLACEMENT_VALUE
                
            print(f"  Actual: {all_targets[i][0]:.2f}, Predicted: {pred_val:.2f}")

    # *** NEW: Return predictions for saving ***
    return all_preds, all_targets


def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load and process data
    data = load_and_preprocess_data(FILE_PATH)
    if data is None:
        return
        
    # 2. Create sequences
    print(f"Creating sequences with length {SEQUENCE_LENGTH}...")
    X_train_seq, y_train_seq, _ = create_sequences(
        data['train']['features'], 
        data['train']['target'], 
        data['train']['run_num'],
        data['train']['seq_num'], # *** NEW ***
        SEQUENCE_LENGTH
    )
    # *** NEW: Get test_seq_info ***
    X_test_seq, y_test_seq, test_seq_info = create_sequences(
        data['test']['features'], 
        data['test']['target'], 
        data['test']['run_num'],
        data['test']['seq_num'], # *** NEW ***
        SEQUENCE_LENGTH
    )

    print(f"Training sequences: X shape {X_train_seq.shape}, y shape {y_train_seq.shape}")
    print(f"Test sequences: X shape {X_test_seq.shape}, y shape {y_test_seq.shape}")

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print("ERROR: No sequences created. Check SEQUENCE_LENGTH or user session lengths.", file=sys.stderr)
        return

    # 3. Create DataLoaders
    train_dataset = RSRPSequenceDataset(X_train_seq, y_train_seq)
    test_dataset = RSRPSequenceDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize model, loss, optimizer
    # *** NEW: Use GRUModel ***
    model = GRUModel(
        input_size=data['num_features'],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    ).to(DEVICE)
    
    # *** NEW: Define model save path ***
    model_save_path = "gru_rsrp_forecaster.pth"
    
    print("\nModel Initialized:")
    print(model)
    print(f"Number of features: {data['num_features']}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    # 5. Train or Load Model
    # *** NEW: Check if model already exists ***
    if os.path.exists(model_save_path):
        print(f"\nLoading existing model from {model_save_path}...")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load model ({e}). Retraining...")
            # If loading fails, train from scratch
            train_model(model, train_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE)
            # 7. *** NEW: Save Model ***
            print(f"\nSaving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
    else:
        # 5. Train
        print(f"\nNo model found at {model_save_path}. Starting training...")
        train_model(model, train_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE)
        # 7. *** NEW: Save Model ***
        print(f"\nSaving model to {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
    
    # 6. Evaluate
    # *** NEW: Get predictions back for saving ***
    all_preds, all_targets = evaluate_model(model, test_loader, data['scaler_target'], DEVICE)

    # 7. *** REMOVED: Model saving moved to after training ***
    
    # 8. *** NEW: Save Results to CSV ***
    print("Saving results to CSV...")
    try:
        # Create the results dataframe
        results_df = test_seq_info.copy() # This has run_num and seq_num
        results_df['actual_nr_ssRsrp'] = all_targets.flatten()
        results_df['predicted_nr_ssRsrp'] = all_preds.flatten()
        results_df['error'] = results_df['actual_nr_ssRsrp'] - results_df['predicted_nr_ssRsrp']
        
        # Load original df to get context features
        df_orig = pd.read_csv(
            FILE_PATH, 
            usecols=['run_num', 'seq_num'] + ALL_CONTEXT_FEATURES
        )
        
        # Merge to add context features to the results
        results_df = pd.merge(
            results_df, 
            df_orig, 
            on=['run_num', 'seq_num'], 
            how='left'
        )
        
        # Re-order columns for clarity
        cols_ordered = ['run_num', 'seq_num'] + \
                       ALL_CONTEXT_FEATURES + \
                       ['actual_nr_ssRsrp', 'predicted_nr_ssRsrp', 'error']
        
        # Filter for any columns that might be missing if df_orig didn't have them
        cols_final = [col for col in cols_ordered if col in results_df.columns]
        results_df = results_df[cols_final]

        results_save_path = "gru_rsrp_forecaster_results.csv"
        results_df.to_csv(results_save_path, index=False)
        print(f"Results saved to {results_save_path}")

    except Exception as e:
        print(f"Error saving results to CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

#   Actual: -140.00, Predicted: -137.75
#   Actual: -92.00, Predicted: -96.18
#   Actual: -140.00, Predicted: -138.40
#   Actual: -89.00, Predicted: -91.02
#   Actual: -104.00, Predicted: -108.10
#   Actual: -92.00, Predicted: -91.23
#   Actual: -87.00, Predicted: -92.81
#   Actual: -81.00, Predicted: -82.54
#   Actual: -140.00, Predicted: -140.00
#   Actual: -79.00, Predicted: -78.59