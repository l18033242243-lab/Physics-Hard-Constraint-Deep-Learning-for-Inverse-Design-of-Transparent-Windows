import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import logging


log_filename = "training_tandem.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()])


N_LAYERS = 13            
N_MATERIALS = 4          
INPUT_DIM_SPECTRA = 502  
OUTPUT_DIM_STRUCT = 65   

D_MIN = 20.0             
D_MAX = 320.0            
TARGET_LAMBDA_C = 520.0  

BATCH_SIZE = 8192        
EPOCHS = 1500           
LEARNING_RATE_INV = 3e-4 


HARD_PHASE_START_EPOCH = int(EPOCHS * 1)


INIT_TEMPERATURE = 3   
MIN_TEMPERATURE = 0.75    
TEMPERATURE_DECAY = (MIN_TEMPERATURE / INIT_TEMPERATURE) ** (1 / max(1, EPOCHS))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



nk_file = "nk.txt"
if not os.path.exists(nk_file):
    raise FileNotFoundError(f"cannot find {nk_file}")

nk_data = np.loadtxt(nk_file, skiprows=1)
wl_np = nk_data[:, 0]
n_np = nk_data[:, 1:1+N_MATERIALS*2:2]
target_idx = np.argmin(np.abs(wl_np - TARGET_LAMBDA_C))
MATERIAL_N_580 = n_np[target_idx, :]


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x) 
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity 
        out = self.act(out)
        return out

class ForwardResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForwardResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(512, 1024),   
            ResidualBlock(1024, 1024),  
            ResidualBlock(1024, 1024),  
            ResidualBlock(1024, 1024)   
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.Sigmoid() 
        )
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


class InverseModelConstrained(nn.Module):

    def __init__(self, temperature=1.0, hidden_dims=[1024, 2048, 1024, 1024], dropout_rate=0.0):
        super().__init__()
        self.temperature = temperature
        
        layers = []
        curr_dim = INPUT_DIM_SPECTRA
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU()) 
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, N_LAYERS * N_MATERIALS))
        self.net = nn.Sequential(*layers)
        

        real_d = [TARGET_LAMBDA_C / (2.0 * n) for n in MATERIAL_N_580]
        norm_d = [(d - D_MIN) / (D_MAX - D_MIN) for d in real_d]
        
        c_mat_left = torch.eye(N_MATERIALS)
        c_mat_right = torch.tensor(norm_d, dtype=torch.float32).unsqueeze(1)
        constraint_matrix = torch.cat([c_mat_left, c_mat_right], dim=1)
        self.register_buffer('constraint_matrix', constraint_matrix)
        
    def forward(self, x, force_hard=False):
        logits_mat = self.net(x).view(-1, N_LAYERS, N_MATERIALS)
        
        if self.training:
            probs_mat = F.gumbel_softmax(logits_mat, tau=self.temperature, hard=force_hard, dim=-1)
        else:
            idx = torch.argmax(logits_mat, dim=-1, keepdim=True)
            probs_mat = torch.zeros_like(logits_mat).scatter_(-1, idx, 1.0)
            
        struct_5D = torch.matmul(probs_mat, self.constraint_matrix)
        struct_pred = struct_5D.view(-1, OUTPUT_DIM_STRUCT)
        return struct_pred


def gpu_batch_iterator(X, batch_size, shuffle=True):

    N = X.size(0)
    perm = torch.randperm(N, device=device) if shuffle else torch.arange(N, device=device)
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield X[idx]


def main():
    target_data_file = 'Inverse_model_train_dataset.npy'
    if not os.path.exists(target_data_file):
        raise FileNotFoundError(f"cannot find {target_data_file}")
        
    data_full = np.load(target_data_file)
    
    total_samples = data_full.shape[0]
    val_size = int(total_samples * 0.1)
    train_size = total_samples - val_size
    
    spectra_train = data_full[:train_size, :].astype(np.float32)
    spectra_val = data_full[train_size:, :].astype(np.float32)
    
    X_train_gpu = torch.tensor(spectra_train, device=device)
    X_val_gpu = torch.tensor(spectra_val, device=device)

    

    model_Forward = ForwardResNet(OUTPUT_DIM_STRUCT, INPUT_DIM_SPECTRA).to(device)
    forward_model_path = 'Forward_surrogate_model_13layers_with_thickness.pth' 
    
    if not os.path.exists(forward_model_path):
        raise FileNotFoundError(f"cannot find {forward_model_path}")
        
    model_Forward.load_state_dict(torch.load(forward_model_path, map_location=device))
    model_Forward.eval()
    for param in model_Forward.parameters(): 
        param.requires_grad = False

    

    model_Inverse = InverseModelConstrained(
        temperature=INIT_TEMPERATURE, 
        hidden_dims=[1024, 2048, 1024, 1024], 
        dropout_rate=0
    ).to(device)
    
    optimizer = optim.Adam(model_Inverse.parameters(), lr=LEARNING_RATE_INV)
    criterion = nn.MSELoss() 
    

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4
    )

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):

        curr_temp = INIT_TEMPERATURE * (TEMPERATURE_DECAY ** epoch)
        model_Inverse.temperature = curr_temp

        is_hard_phase = (epoch >= HARD_PHASE_START_EPOCH)

        model_Inverse.train()
        running_loss = 0.0
        
        for target_spectra in gpu_batch_iterator(X_train_gpu, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            

            struct_pred = model_Inverse(target_spectra, force_hard=is_hard_phase)
            pred_spectra = model_Forward(struct_pred)
            

            loss = criterion(pred_spectra, target_spectra)
            
            loss.backward()  
            torch.nn.utils.clip_grad_norm_(model_Inverse.parameters(), max_norm=1.0)
            optimizer.step()
            
            bs = target_spectra.size(0)
            running_loss += loss.item() * bs


        model_Inverse.eval() 
        running_val_mse = 0.0
        with torch.no_grad():
            for target_v in gpu_batch_iterator(X_val_gpu, BATCH_SIZE * 4, shuffle=False):
                struct_v = model_Inverse(target_v, force_hard=True)
                pred_v = model_Forward(struct_v)
                running_val_mse += criterion(pred_v, target_v).item() * target_v.size(0)

        scheduler.step()
        

        if (epoch + 1) % 1 == 0 or epoch == 0:
            avg_mse = running_loss / train_size
            avg_val_mse = running_val_mse / val_size
            current_lr = optimizer.param_groups[0]['lr']
            
            phase_str = "HARD" if is_hard_phase else "SOFT"
            
            elapsed = time.time() - start_time
            logging.info(f"[Epoch {epoch+1:4d}/{EPOCHS}] [{phase_str}] | "
                         f"LR: {current_lr:.2e} | Temp: {curr_temp:.3f} | "
                         f"Train MSE: {avg_mse:.5f} | Val MSE: {avg_val_mse:.5f} | "
                         f"Time: {elapsed:.1f}s")
            
            start_time = time.time()
            
            if avg_val_mse < best_val_loss:
                best_val_loss = avg_val_mse
                torch.save(model_Inverse.state_dict(), 'net_inverse.pth')

if __name__ == "__main__":
    main()