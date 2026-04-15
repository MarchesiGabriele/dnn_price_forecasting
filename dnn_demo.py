#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist


# In[ ]:


args = {
    'seed': 20,
    'region': "BE",
    'predict_horizon': 24,
    'recalibration_shift_days': 100,
    'val_ratio': 0.2, # % of the train data to use for validation

    # Data parameters
    'train_start': '2019-01-01',
    'train_end': '2023-10-01',
    'test_start': '2023-10-01', # included
    'test_end': '2024-10-01', # esluded

    # DNN parameters
    'target_quantiles': [i/100 for i in range(1, 100)],
    'perform_RevIN': True, # only true allowed for now
    'hidden_size': 128,  
    'hidden_layers': 2,
    'activation_fn': 'ReLU', # currently only ReLU is allowed
    'dropout_rate': 0.1,
    'mixture_components': 3,
    'context_window_days': [-1,-2,-7], # days indexes before the forecast horizon
    'full_history_hours': 168, # total hours to use for RevIN stats
    'epochs': 800,
    'patience': 10,
    'learning_rate': 5e-4,
    'batch_size': 128,
    'num_workers': 4,
}


# In[ ]:


# SETUP DEVICE AND FOLDERS

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Create checkpoint and log folders
os.makedirs(f'log_dir', exist_ok=True)
os.makedirs(f'checkpoints', exist_ok=True)
os.makedirs(f'results', exist_ok=True)


# In[ ]:


# LOADING DATA

df_raw = pd.read_csv("./BE/df_full_scaled.csv")
feature_cols = list(df_raw.columns)
feature_cols.remove('TARG__target_scaled')
feature_cols.remove('date')
df_raw['date'] = pd.to_datetime(df_raw['date'])

denorm_params = pd.read_csv("./BE/df_target_denorm_params.csv")
denorm_params['date'] = pd.to_datetime(denorm_params['date'])

cons_cols = [col for col in feature_cols if 'CONS' in col]
futu_cols = [col for col in feature_cols if 'CONS' not in col]
args['input_data_shape'] = len(args['context_window_days'])*24 + (len(futu_cols)*args['predict_horizon'] + len(cons_cols))


# In[ ]:


# QUANTILE REGRESSION (DNN)

class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.args = args

        self.input_layer = nn.Sequential(
            nn.Linear(self.args['input_data_shape'], self.args['hidden_size']), 
            nn.ReLU(),
            nn.Dropout(p=self.args['dropout_rate'])
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.args['hidden_size'], self.args['hidden_size']),
                nn.ReLU(),
                nn.Dropout(p=self.args['dropout_rate']),
            ) for _ in range(self.args['hidden_layers'] - 1)
        ])

        self.out_features = len(self.args['target_quantiles'])
        self.output_layer = nn.Linear(self.args['hidden_size'], self.out_features*self.args['predict_horizon'])

    def forward(self, past_target, future_features, target_mean, target_std): 
        # past_target: (batch, len(context_window_days)*24)
        # future_features: (batch, num_normal*24 + num_cons)

        # Inputs are already normalized by the dataset
        x = torch.cat([past_target, future_features], dim=1) # (batch, total_input_features)

        assert x.shape[1] == self.args['input_data_shape'], f"Input shape mismatch: expected {self.args['input_data_shape']}, got {x.shape[1]}"

        x = self.input_layer(x) 

        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)

        # Reshape to (batch, horizon, quantiles)
        out = out.view(-1, self.args['predict_horizon'], self.out_features)

        # Denormalize + reshape for broadcasting
        out = out * target_std.view(-1, 1, 1) + target_mean.view(-1, 1, 1)

        # fix quantiles crossing
        out, _ = torch.sort(out, dim=-1) 
        return out 


# In[ ]:


class PinballLoss(torch.nn.Module):
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_true, y_pred):
        # y_true: (batch, horizon)
        # y_pred: (batch, horizon, quantiles)

        # Ensure quantiles are on the same device as input
        if not hasattr(self, 'q_tensor') or self.q_tensor.device != y_true.device:
            self.q_tensor = torch.tensor(self.quantiles, dtype=y_true.dtype, device=y_true.device).view(1, 1, -1)

        # Broadcast y_true to (batch, horizon, 1) to match y_pred's quantile dim
        error = y_true.unsqueeze(-1) - y_pred

        # Vectorized Pinball Loss: max(q * error, (q - 1) * error)
        # q_tensor broadcasts to (1, 1, n_quantiles)
        loss = torch.mean(torch.maximum(self.q_tensor * error, (self.q_tensor - 1) * error))

        return loss

    def get_config(self):
        return {
            "num_quantiles": self.quantiles,
        }


class DistributionNLLLoss(torch.nn.Module):
    def forward(self, y_true, pred_dist):
        # y_true: (batch, horizon)
        # pred_dist: torch.distributions.Distribution with batch_shape (batch, horizon)
        return -pred_dist.log_prob(y_true).mean()


# In[ ]:


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, target: str, context_window_days: list, full_history_hours: int, prediction_horizon: int, daily_aligned: bool = False):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target = target
        self.context_window_days = context_window_days
        self.full_history_hours = full_history_hours
        self.prediction_horizon = prediction_horizon
        self.all_features_unnorm = self.df[self.feature_cols].values.astype(np.float32)
        self.all_target_unnorm = self.df[self.target].values.astype(np.float32)
        self.dates = self.df['date'].values
        self.static_features_idx = [self.feature_cols.index(col) for col in self.feature_cols if 'CONS' in col]
        self.dynamic_features_idx = [self.feature_cols.index(col) for col in self.feature_cols if 'CONS' not in col]

        # Pre-compute valid indices: only those where prediction starts at midnight
        total_window = self.full_history_hours + self.prediction_horizon
        all_count = max(0, len(self.df) - total_window + 1)
        if daily_aligned:
            self.valid_indices = [
                i for i in range(all_count)
                if pd.to_datetime(self.dates[i + self.full_history_hours]).hour == 0
            ]
        else:
            self.valid_indices = list(range(all_count))


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        history window: [idx : idx + full_history_hours]
        prediction horizon: [idx + full_history_hours : idx + full_history_hours + prediction_horizon]

        - Normalization stats are computed on the past window of lenght: [full_history_hours] from settings
        - Normalization also affects the time encoding variables
        """
        # Map external idx to internal data index
        idx = self.valid_indices[idx]
        past_target_window_unnorm = self.all_target_unnorm[idx : idx + self.full_history_hours]
        past_features_window_unnorm = self.all_features_unnorm[idx : idx + self.full_history_hours]

        # Calculate RevIN stats
        past_target_mean = past_target_window_unnorm.mean()
        past_target_std = past_target_window_unnorm.std() + 1e-5
        past_features_mean = past_features_window_unnorm.mean(axis=0)
        past_features_std = past_features_window_unnorm.std(axis=0) + 1e-5

        # Normalize target for the full window
        full_window_target_unnorm = self.all_target_unnorm[idx : idx + self.full_history_hours + self.prediction_horizon]
        full_window_target_norm = (full_window_target_unnorm - past_target_mean) / past_target_std

        # Normalize all features
        full_window_features_norm = (self.all_features_unnorm[idx : idx + self.full_history_hours + self.prediction_horizon] - past_features_mean) / past_features_std

        # 1. Past Target: select specific days from the normalized history
        past_target_input_list = []
        for day_offset in self.context_window_days:
            start_in_window = self.full_history_hours + (day_offset * 24)
            end_in_window = start_in_window + 24
            past_target_input_list.append(full_window_target_norm[start_in_window:end_in_window])

        past_target_input_norm = np.concatenate(past_target_input_list) # (len(days)*24,)

        # 2. Future Features: prediction horizon part of full_window_features_norm
        future_features_window_norm = full_window_features_norm[self.full_history_hours : self.full_history_hours + self.prediction_horizon]

        # Dynamic features (all 24h)
        future_dynamic_features_norm = future_features_window_norm[:, self.dynamic_features_idx].flatten() 
        # Static features (only 1st value)
        future_static_features_norm = future_features_window_norm[0, self.static_features_idx]

        future_features_input_norm = np.concatenate([future_dynamic_features_norm, future_static_features_norm])

        # 3. Future Target (raw/unnormalized)
        future_target_label_unnorm = self.all_target_unnorm[idx + self.full_history_hours : idx + self.full_history_hours + self.prediction_horizon]

        return (torch.from_numpy(past_target_input_norm), 
                torch.from_numpy(future_features_input_norm), 
                torch.from_numpy(future_target_label_unnorm), 
                torch.tensor(past_target_mean, dtype=torch.float32), 
                torch.tensor(past_target_std, dtype=torch.float32))


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.mean(np.abs(pred - true)))


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_picp(y_true: np.ndarray, pred_quantiles: np.ndarray, quantiles: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    pred_quantiles = np.asarray(pred_quantiles, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if pred_quantiles.ndim != 3:
        raise ValueError(f"pred_quantiles must have shape (N, H, Q), got {pred_quantiles.shape}")

    target_low = (1.0 - alpha) / 2.0
    target_high = 1.0 - target_low
    idx_low = int(np.abs(quantiles - target_low).argmin())
    idx_high = int(np.abs(quantiles - target_high).argmin())

    lower = pred_quantiles[:, :, idx_low]
    upper = pred_quantiles[:, :, idx_high]
    covered = np.logical_and(lower <= y_true, y_true <= upper)
    return float(np.mean(covered))


def compute_crps_quantile(labels: np.ndarray, pred_quantiles: np.ndarray, quantiles: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    pred_quantiles = np.asarray(pred_quantiles, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    if pred_quantiles.ndim != 3:
        raise ValueError(f"pred_quantiles must have shape (N, H, Q), got {pred_quantiles.shape}")
    if quantiles.ndim != 1:
        raise ValueError(f"quantiles must be a 1D array, got {quantiles.shape}")

    labels_flat = labels.reshape(-1)
    pred_flat = pred_quantiles.reshape(-1, pred_quantiles.shape[-1])
    errors = labels_flat[:, None] - pred_flat
    quantiles_row = quantiles[None, :]
    loss = np.maximum(quantiles_row * errors, (quantiles_row - 1.0) * errors)
    return float(np.mean(loss))


def _sample_normal_quantiles(
    loc: np.ndarray,
    scale: np.ndarray,
    quantiles: np.ndarray,
    num_samples: int = 1000,
    seed: int = 20,
) -> np.ndarray:
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    if np.any(scale <= 0):
        # Guard against numerical issues in saved CSVs.
        scale = np.maximum(scale, 1e-6)

    rng = np.random.default_rng(seed)
    loc_tensor = torch.as_tensor(loc, dtype=torch.float64)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float64)
    normal_dist = dist.Normal(loc=loc_tensor, scale=scale_tensor)

    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    draws = normal_dist.sample((num_samples,)).cpu().numpy().T
    return np.quantile(draws, quantiles, axis=1).T


def _sample_mixnormal_quantiles(
    loc: np.ndarray,
    scale: np.ndarray,
    logits: np.ndarray,
    quantiles: np.ndarray,
    num_samples: int = 1000,
    seed: int = 20,
) -> np.ndarray:
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    if loc.ndim != 2 or scale.ndim != 2 or logits.ndim != 2:
        raise ValueError("loc, scale and logits must have shape (N, K)")
    if loc.shape != scale.shape or loc.shape != logits.shape:
        raise ValueError("loc, scale and logits must share the same shape")

    if np.any(scale <= 0):
        scale = np.maximum(scale, 1e-6)

    rng = np.random.default_rng(seed)
    loc_tensor = torch.as_tensor(loc, dtype=torch.float64)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float64)
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)

    mix = dist.Categorical(logits=logits_tensor)
    comp = dist.Normal(loc=loc_tensor, scale=scale_tensor)
    mix_dist = dist.MixtureSameFamily(mix, comp)

    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    draws = mix_dist.sample((num_samples,)).cpu().numpy().T
    return np.quantile(draws, quantiles, axis=1).T


def load_results_quantiles(
    model_type: str,
    quantiles: np.ndarray | None = None,
    num_samples: int = 1000,
    seed: int | None = None,
):
    pred_path = os.path.join("results", f"{model_type}_pred.csv")
    target_path = os.path.join("results", f"{model_type}_target.csv")

    preds_df = pd.read_csv(pred_path, parse_dates=["date"]).set_index("date").sort_index()
    target_df = pd.read_csv(target_path, parse_dates=["date"]).set_index("date").sort_index()

    common_index = target_df.index.intersection(preds_df.index)
    preds_df = preds_df.loc[common_index]
    target_df = target_df.loc[common_index]
    y_true = target_df["target"].to_numpy(dtype=np.float64).reshape(-1, 1)

    if model_type == "DNN":
        quantile_columns = [col for col in preds_df.columns if col.startswith("q_")]
        quantiles = np.array([float(col.split("_", 1)[1]) for col in quantile_columns], dtype=np.float64)
        order = np.argsort(quantiles)
        quantiles = quantiles[order]
        pred_quantiles = preds_df[quantile_columns].to_numpy(dtype=np.float64)[:, order]
    elif model_type == "DNN_N":
        quantiles = np.sort(np.asarray(args["target_quantiles"] if quantiles is None else quantiles, dtype=np.float64))

        pred_quantiles = _sample_normal_quantiles(
            loc=preds_df["loc"].to_numpy(dtype=np.float64),
            scale=preds_df["scale"].to_numpy(dtype=np.float64),
            quantiles=quantiles,
            num_samples=num_samples,
            seed=args["seed"] if seed is None else seed,
        )
    elif model_type == "DNN_MIXN":
        num_components = int(args["mixture_components"])
        loc_columns = [f"loc_{i+1}" for i in range(num_components)]
        scale_columns = [f"scale_{i+1}" for i in range(num_components)]
        logit_columns = [f"logit_{i+1}" for i in range(num_components)]

        quantiles = np.sort(np.asarray(args["target_quantiles"] if quantiles is None else quantiles, dtype=np.float64))

        pred_quantiles = _sample_mixnormal_quantiles(
            loc=preds_df[loc_columns].to_numpy(dtype=np.float64),
            scale=preds_df[scale_columns].to_numpy(dtype=np.float64),
            logits=preds_df[logit_columns].to_numpy(dtype=np.float64),
            quantiles=quantiles,
            num_samples=num_samples,
            seed=args["seed"] if seed is None else seed,
        )

    return y_true, pred_quantiles[:, None, :], quantiles, common_index


def compute_results_metrics(
    model_type: str,
    alpha: float = 0.90,
    quantiles: np.ndarray | None = None,
    num_samples: int = 1000,
    seed: int | None = None,
    point_quantile: float = 0.50,
):
    y_true, pred_quantiles, quantiles, common_index = load_results_quantiles(
        model_type=model_type,
        quantiles=quantiles,
        num_samples=num_samples,
        seed=seed,
    )

    point_idx = int(np.abs(quantiles - point_quantile).argmin())
    point_pred = pred_quantiles[:, 0, point_idx]
    y_true_flat = y_true[:, 0]

    metrics = {
        "MAE": MAE(point_pred, y_true_flat),
        "RMSE": RMSE(point_pred, y_true_flat),
    }

    for picp_alpha in sorted({0.50, 0.90, 0.98, float(alpha)}):
        metrics[f"PICP_{int(round(picp_alpha * 100))}"] = compute_picp(
            y_true,
            pred_quantiles,
            quantiles,
            picp_alpha,
        )

    metrics["CRPS"] = compute_crps_quantile(y_true, pred_quantiles, quantiles)

    print(f"\nMetrics for {model_type} on {len(common_index)} timestamps:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    return metrics


# In[ ]:


# TRAINING FUNCTION

def train_on_split(train_raw, suffix, feature_cols, model_class, model_type):
    best_val_loss = float('inf')
    model = model_class(args)
    model.to(device)
    if model_type == "DNN":
        loss_fn = PinballLoss(args['target_quantiles']).to(device)
    elif model_type in {"DNN_N", "DNN_MIXN"}:
        loss_fn = DistributionNLLLoss().to(device)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    run_name = f"{model_type}_{suffix}"
    checkpoint_path = os.path.join("checkpoints", f"best_model_{run_name}.pth")
    run_log_dir = os.path.join("log_dir", model_type, suffix)
    writer = SummaryWriter(log_dir=run_log_dir)
    print(f"TensorBoard logs: {run_log_dir}")

    # Build ONE dataset from all train data, with daily alignment (1 sample per day)
    full_dataset = CustomDataset(
        train_raw,
        feature_cols, 
        'TARG__target_scaled',
        context_window_days=args['context_window_days'],
        full_history_hours=args['full_history_hours'],
        prediction_horizon=args['predict_horizon'],
        daily_aligned=True  # always daily-aligned
    )

    n_samples = len(full_dataset)
    n_val = int(n_samples * args['val_ratio'])
    n_train = n_samples - n_val

    # Shuffle sample indices BEFORE splitting (like the paper)
    all_indices = list(range(n_samples))
    np.random.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    print(f"Train/Val split: {n_train} train samples, {n_val} val samples (total: {n_samples})")

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset, 
        batch_size=args['batch_size'], 
        shuffle=True, 
        num_workers=args['num_workers'], 
        pin_memory=True, 
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_subset, 
        batch_size=args['batch_size'], 
        num_workers=args['num_workers'], 
        pin_memory=True, 
        persistent_workers=False
    )

    patience_counter = 0
    for epoch in range(args['epochs']):
        # train
        model.train()
        train_losses = []
        for (past_target, future_features, future_target, target_mean, target_std) in train_loader:
            past_target = past_target.to(device)
            future_features = future_features.to(device)
            future_target = future_target.to(device)
            target_mean = target_mean.to(device)
            target_std = target_std.to(device)
            optimizer.zero_grad()
            output = model(past_target, future_features, target_mean, target_std)
            loss = loss_fn(future_target, output)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation 
        model.eval()
        val_losses = [] 
        with torch.no_grad():
            for (past_target, future_features, future_target, target_mean, target_std) in val_loader:
                past_target = past_target.to(device)
                future_features = future_features.to(device)
                future_target = future_target.to(device)
                target_mean = target_mean.to(device)
                target_std = target_std.to(device)
                output = model(past_target, future_features, target_mean, target_std)
                loss = loss_fn(future_target, output)
                val_losses.append(loss.item())

        print(f"Epoch {epoch+1} - Train Loss: {np.mean(train_losses):.4f} - Val Loss: {np.mean(val_losses):.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
        writer.add_scalar('Loss/val', np.mean(val_losses), epoch)

        if np.mean(val_losses) < best_val_loss: 
            print(f"New best validation loss: {np.mean(val_losses):.4f}, saving model...")
            best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 0: print(f"Patience counter: {patience_counter} out of {args['patience']}")
        if patience_counter >= args['patience']:
            print(f"Patience reached, stopping training...")
            break

    # Explicitly delete loaders and datasets to trigger worker shutdown and memory release
    writer.close()
    del train_loader, val_loader, train_subset, val_subset, full_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# In[ ]:


# WALK FORWARD EXPERIMENT

def walk_forward_experiment(model_class, model_type):
    train_raw = df_raw[(df_raw['date'] >= pd.to_datetime(args['train_start'])) & (df_raw['date'] < pd.to_datetime(args['train_end']))]
    test_raw = df_raw[(df_raw['date'] >= pd.to_datetime(args['test_start'])) & (df_raw['date'] < pd.to_datetime(args['test_end']))]

    print(f"Loaded data from {df_raw['date'].iloc[0]} to {df_raw['date'].iloc[-1]}")

    preds_list = []
    targets_list = []

    day_count = 0 
    while len(test_raw) > 0: 
        print(f"\n{'='*60}\nDay {day_count}: {test_raw['date'].iloc[0].strftime('%Y-%m-%d')}")

        # Check if we need to retrain
        if day_count % args['recalibration_shift_days'] == 0:
            run_suffix = f"week{day_count // args['recalibration_shift_days']}_{args['region']}_{args['seed']}"
            print(f"Retraining - MarketRegion: {args['region']} - Seed: {args['seed']} - Retrain split: {day_count // args['recalibration_shift_days']} / {int(365/args['recalibration_shift_days'])}")
            print("First day of train data: ", train_raw['date'].iloc[0])
            print("Last day of train data: ", train_raw['date'].iloc[-1])
            print("First day of test data: ", test_raw['date'].iloc[0])
            print("Last day of test data: ", test_raw['date'].iloc[-1])
            print(f"Train data shape: {train_raw.shape}")
            print(f"Test data shape: {test_raw.shape}")

            train_on_split(
                train_raw=train_raw,
                suffix=run_suffix,
                feature_cols=feature_cols,
                model_class=model_class,
                model_type=model_type
            )

        # Evaluation
        run_suffix = f"week{day_count // args['recalibration_shift_days']}_{args['region']}_{args['seed']}"
        checkpoint_path = os.path.join("checkpoints", f"best_model_{model_type}_{run_suffix}.pth")
        model = model_class(args) 
        model.to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        test_dataset = CustomDataset(
            pd.concat([train_raw, test_raw.iloc[:args['predict_horizon']]]),
            feature_cols,
            'TARG__target_scaled',
            context_window_days=args['context_window_days'],
            full_history_hours=args['full_history_hours'],
            prediction_horizon=args['predict_horizon'],
        )
        # We want the last sample of the test_dataset which corresponds to predicting the current test_raw horizon
        past_target, future_features, future_target, target_mean, target_std = test_dataset[len(test_dataset)-1]

        print("First sample of test interval: ", test_dataset.df['date'].iloc[test_dataset.valid_indices[-1] + test_dataset.full_history_hours])
        print("Last sample of test interval: ", test_dataset.df['date'].iloc[test_dataset.valid_indices[-1] + test_dataset.full_history_hours + test_dataset.prediction_horizon - 1])
        assert test_dataset.df['date'].iloc[test_dataset.valid_indices[-1] + test_dataset.full_history_hours] == test_raw['date'].iloc[0], "First sample of test interval does not match first sample of test_raw"
        assert test_dataset.df['date'].iloc[test_dataset.valid_indices[-1] + test_dataset.full_history_hours + test_dataset.prediction_horizon - 1] == test_raw['date'].iloc[args['predict_horizon'] - 1], "Last sample of test interval does not match last sample of test_raw"

        with torch.no_grad():
            past_target = past_target.unsqueeze(0).to(device)
            future_features = future_features.unsqueeze(0).to(device)
            future_target = future_target.to(device)
            target_mean = target_mean.to(device)
            target_std = target_std.to(device)
            if model_type == "DNN":
                output = model(past_target, future_features, target_mean, target_std).squeeze(0)
            elif model_type == "DNN_N":
                pred_dist = model(
                    past_target,
                    future_features,
                    target_mean,
                    target_std,
                )
                affine = pred_dist.transforms[0]
                loc = pred_dist.base_dist.loc * affine.scale + affine.loc
                scale = pred_dist.base_dist.scale * affine.scale.abs()
                output = torch.stack([loc, scale], dim=-1).squeeze(0)
            elif model_type == "DNN_MIXN":
                pred_dist = model(
                    past_target,
                    future_features,
                    target_mean,
                    target_std,
                )
                affine = pred_dist.transforms[0]
                component_dist = pred_dist.base_dist.component_distribution
                mixture_dist = pred_dist.base_dist.mixture_distribution
                loc = component_dist.loc * affine.scale.unsqueeze(-1) + affine.loc.unsqueeze(-1)
                scale = component_dist.scale * affine.scale.abs().unsqueeze(-1)
                logits = mixture_dist.logits[:, 0, :]
                horizon = loc.shape[1]
                logits = logits.unsqueeze(1).expand(-1, horizon, -1)
                output = torch.cat(
                    [loc, scale, logits],
                    dim=-1
                ).squeeze(0)
            else:
                raise ValueError(f"Model type {model_type} not supported")

        # If we have less than a full horizon left in test_raw, 
        # we should only take the entries that correspond to the test_raw entries.
        L = min(len(test_raw), args['predict_horizon'])
        if L < args['predict_horizon']:
            output = output[:L] # take the first L entries (corresponding to the start of the horizon)
            future_target = future_target[:L]

        # Denormalize if using scaled data
        test_dates = test_raw['date'].iloc[:L].values
        denorm_slice = denorm_params.set_index('date').loc[test_dates]
        p0 = torch.tensor(denorm_slice['CONS__TARG__target_trasf_p0'].values, dtype=output.dtype).to(device)
        p1 = torch.tensor(denorm_slice['CONS__TARG__target_trasf_p1'].values, dtype=output.dtype).to(device)
        if model_type == "DNN":
            output = output * p1.unsqueeze(1) + p0.unsqueeze(1)
            future_target = future_target * p1 + p0
        elif model_type == "DNN_N":
            # mu_y    = m + s * mu_z
            # sigma_y = s * sigma_z
            # output shape (L, 2) where first column is mu_y and second column is sigma_y
            output[:, 0] = output[:, 0] * p1 + p0
            output[:, 1] = output[:, 1] * p1.abs()
            future_target = future_target * p1 + p0
        elif model_type == "DNN_MIXN":
            num_components = int(args["mixture_components"])
            output[:, :num_components] = output[:, :num_components] * p1.unsqueeze(1) + p0.unsqueeze(1)
            output[:, num_components:2 * num_components] = (
                output[:, num_components:2 * num_components] * p1.abs().unsqueeze(1)
            )
            future_target = future_target * p1 + p0
        else:
            raise ValueError(f"Model type {model_type} not supported")

        output_np = output.detach().cpu().numpy()
        future_target_np = future_target.detach().cpu().numpy()
        test_dates_pd = pd.to_datetime(test_dates)

        if model_type == "DNN":
            pred_columns = [f"q_{q:.2f}" for q in args['target_quantiles']]
        elif model_type == "DNN_N":
            pred_columns = ["loc", "scale"]
        elif model_type == "DNN_MIXN":
            num_components = int(args["mixture_components"])
            pred_columns = (
                [f"loc_{i+1}" for i in range(num_components)] +
                [f"scale_{i+1}" for i in range(num_components)] +
                [f"logit_{i+1}" for i in range(num_components)]
            )
        else:
            raise ValueError(f"Model type {model_type} not supported")

        preds_list.append(pd.DataFrame(output_np, columns=pred_columns, index=test_dates_pd))
        targets_list.append(pd.DataFrame({"target": future_target_np}, index=test_dates_pd))

        # SHIFT DATASETS
        train_raw = pd.concat([train_raw, test_raw.iloc[:args['predict_horizon']]]) # concatenate the first args['predict_horizon'] days of test to train
        test_raw = test_raw.iloc[args['predict_horizon']:] # remove the first args['predict_horizon'] days of test from test
        train_raw = train_raw.iloc[args['predict_horizon']:] # remove the first args['predict_horizon'] days of train from train
        day_count += 1

    preds_df = pd.concat(preds_list).sort_index()
    target_df = pd.concat(targets_list).sort_index()
    preds_df.index.name = "date"
    target_df.index.name = "date"

    preds_path = os.path.join("results", f"{model_type}_pred.csv")
    target_path = os.path.join("results", f"{model_type}_target.csv")
    preds_df.to_csv(preds_path)
    target_df.to_csv(target_path)

    print(f"Saved predictions to {preds_path}")
    print(f"Saved targets to {target_path}")

    return preds_df, target_df


# In[ ]:


# DISTRIBUTIONAL DNN (DNN_N)

class DNN_N(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.input_layer = nn.Sequential(
            nn.Linear(self.args['input_data_shape'], self.args['hidden_size']),
            nn.ReLU(),
            nn.Dropout(p=self.args['dropout_rate'])
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.args['hidden_size'], self.args['hidden_size']),
                nn.ReLU(),
                nn.Dropout(p=self.args['dropout_rate']),
            ) for _ in range(self.args['hidden_layers'] - 1)
        ])

        self.out_features = 2 # loc, scale
        self.output_layer = nn.Linear(
            self.args['hidden_size'],
            self.out_features * self.args['predict_horizon']
        )

    def forward(self, past_target, future_features, target_mean, target_std):
        x = torch.cat([past_target, future_features], dim=1)

        assert x.shape[1] == self.args['input_data_shape'], (
            f"Input shape mismatch: expected {self.args['input_data_shape']}, got {x.shape[1]}"
        )

        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)

        out = self.output_layer(x)  # (batch, 2 * horizon)

        # Match the TensorFlow implementation:
        # [loc_1, ..., loc_H, raw_scale_1, ..., raw_scale_H]
        horizon = self.args['predict_horizon']
        loc_norm = out[:, :horizon]            # (batch, horizon)
        raw_scale = out[:, horizon:]           # (batch, horizon)

        # Come in TF: 1e-3 + 3 * softplus(...)
        scale_norm = 1e-3 + 3.0 * F.softplus(raw_scale)

        target_mean = target_mean.view(-1, 1)
        target_std = target_std.view(-1, 1)

        # Distribuzione nello spazio normalizzato
        base_dist = dist.Normal(loc=loc_norm, scale=scale_norm)

        # Trasformazione affine verso lo spazio originale:
        # y = x * target_std + target_mean
        transformed_dist = dist.TransformedDistribution(
            base_dist,
            [dist.transforms.AffineTransform(loc=target_mean, scale=target_std)]
        )
        return transformed_dist


class DNN_MIXN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_components = int(self.args['mixture_components'])

        self.input_layer = nn.Sequential(
            nn.Linear(self.args['input_data_shape'], self.args['hidden_size']),
            nn.ReLU(),
            nn.Dropout(p=self.args['dropout_rate'])
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.args['hidden_size'], self.args['hidden_size']),
                nn.ReLU(),
                nn.Dropout(p=self.args['dropout_rate']),
            ) for _ in range(self.args['hidden_layers'] - 1)
        ])

        self.out_features = 2 * self.args['predict_horizon'] * self.num_components + self.num_components
        self.output_layer = nn.Linear(self.args['hidden_size'], self.out_features)

    def forward(self, past_target, future_features, target_mean, target_std):
        x = torch.cat([past_target, future_features], dim=1)

        assert x.shape[1] == self.args['input_data_shape'], (
            f"Input shape mismatch: expected {self.args['input_data_shape']}, got {x.shape[1]}"
        )

        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)

        out = self.output_layer(x)
        horizon = self.args['predict_horizon']
        comp_block = horizon * self.num_components

        loc_norm = out[:, :comp_block].view(-1, horizon, self.num_components)
        raw_scale = out[:, comp_block:2 * comp_block].view(-1, horizon, self.num_components)
        logits_norm = out[:, 2 * comp_block:]
        scale_norm = 1e-3 + 3.0 * F.softplus(raw_scale)
        target_mean = target_mean.view(-1, 1)
        target_std = target_std.view(-1, 1)

        logits_tiled = logits_norm.view(-1, 1, self.num_components).expand(-1, self.args['predict_horizon'], -1)

        mixture_dist = dist.Categorical(logits=logits_tiled)
        component_dist = dist.Normal(loc=loc_norm, scale=scale_norm)
        base_dist = dist.MixtureSameFamily(mixture_dist, component_dist)

        transformed_dist = dist.TransformedDistribution(
            base_dist,
            [dist.transforms.AffineTransform(loc=target_mean, scale=target_std)]
        )
        return transformed_dist





if __name__ == "__main__":
    # walk_forward_experiment(DNN, "DNN")
    # walk_forward_experiment(DNN_N, "DNN_N")
    walk_forward_experiment(DNN_MIXN, "DNN_MIXN")

    # compute_results_metrics(model_type="DNN")
    # compute_results_metrics(model_type="DNN_N")
    compute_results_metrics(model_type="DNN_MIXN")
