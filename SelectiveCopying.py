import torch
import torch.nn as nn
import torch.nn.functional as F
from data import synthetic_data, vocab_size
import numpy as np
from matplotlib import pyplot as plt

class SelectiveSSM(nn.Module):
    def __init__(self,d_model,d_state,dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_model // 16) 

        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_model, 1)
        self.A_t = nn.Parameter(-torch.exp(A_log))

        self.B_raw = nn.Linear(d_model,d_state,bias = False)
        self.C_raw = nn.Linear(d_model,d_state,bias = False)

        self.delta_raw = nn.Sequential(
                        nn.Linear(d_model, dt_rank, bias=False),
                         nn.Linear(dt_rank, d_model, bias=True)
                        )


    def discretize(self,x_t):
 
        delta_t = F.softplus(self.delta_raw(x_t)).unsqueeze(-1)

        B_t = self.B_raw(x_t).unsqueeze(1)
        C_t = self.C_raw(x_t).unsqueeze(1)

        A_bar_t = torch.exp(self.A_t*delta_t)
        B_bar_t = (1.0 / (delta_t * self.A_t)) * (A_bar_t - 1.0) * (delta_t * B_t)

        return A_bar_t,B_bar_t,C_t


    def forward(self, x_matrix):

        batch,seq_length,_ = x_matrix.shape

        outputs = []   
        H_values = []
        H_prevt = torch.zeros(batch,self.d_model,self.d_state, device=x_matrix.device, dtype=x_matrix.dtype)

        for i in range(0,seq_length):
            x_t = x_matrix[:,i,:]

            A_bar_t,B_bar_t,C_t = self.discretize(x_t)

            H_t = (A_bar_t*H_prevt) + (B_bar_t*(x_t.unsqueeze(-1)))
            Y_t = torch.sum(C_t * H_t, dim=-1)
            outputs.append(Y_t)
            H_values.append(H_t)
            H_prevt = H_t
        H_values = torch.stack(H_values, dim = 1)
        outputs = torch.stack(outputs, dim=1)
        return outputs

class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state, dt_rank: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dt_rank)

    def forward(self, x):
        # pre-norm residual, same pattern as Transformer blocks
        return x + self.ssm(self.norm(x))

class SelectiveCopyingModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, dt_rank: int = 1, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SSMBlock(d_model, d_state, dt_rank) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        
        x = self.embed(token_ids)          # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)                   # (batch, seq_len, d_model)
        x = self.final_norm(x)
        logits = self.output_head(x)       # (batch, seq_len, vocab_size)
        return logits
'''
from data import synthetic_data, vocab_size  
d_model = 64
d_state = 16

model = SelectiveCopyingModel(vocab_size=vocab_size, d_model=d_model, d_state=d_state)

seqs, targets, masks = synthetic_data(batch=4, L=64, K=16)
logits = model(seqs)
'''

def Loss_Function(logits, targets, masks):

    batch_size, seq_len, vocab_size = logits.shape
    K = targets.shape[1]

    answer_logits = logits[masks].view(batch_size, K, vocab_size)  

    
    loss = F.cross_entropy(
        answer_logits.reshape(-1, vocab_size),   
        targets.reshape(-1)                       
    )

    # ---- exact-match accuracy ----
    predictions = answer_logits.argmax(dim=-1)      
    correct_per_slot = (predictions == targets)     
    exact_match = correct_per_slot.all(dim=1).float() 
    exact_match_acc = exact_match.mean()

    return loss, exact_match_acc
'''
loss,acc = Loss_Function(logits=logits,targets=targets,masks=masks)

print(loss,acc)
'''
def train(
    num_steps: int = 8000,
    batch_size: int = 64,
    L: int = 64,
    K: int = 16,
    d_model: int = 64,
    d_state: int = 16,
    vocab_size: int = 32,
    lr: float = 1e-3,
    log_every: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = SelectiveCopyingModel(vocab_size=vocab_size, d_model=d_model, d_state=d_state).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = [ ]
    for step in range(1, num_steps + 1):
       
        seqs, targets, masks = synthetic_data(batch=batch_size, L=L, K=K)
        seqs, targets, masks = seqs.to(device), targets.to(device), masks.to(device)

        
        logits = model(seqs)

        
        loss, acc = Loss_Function(logits, targets, masks)

        losses.append(loss.item())        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.4f} | exact-match acc {acc.item()*100:5.1f}%")

    return model,np.array(losses)


model_new,losses = train(8000,50,10,4,16,8)


t = range(8000)
plt.plot(t,losses)
plt.show()

