import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self,d_model,d_state,dt_rank: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank  

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

        B_t = self.B_raw(x_t).unsqueeze(-1).T
        C_t = self.C_raw(x_t).unsqueeze(-1).T

        A_bar_t = torch.exp(self.A_t*delta_t)
        B_bar_t = (1.0 / (delta_t * self.A_t)) * (A_bar_t - 1.0) * (delta_t * B_t)

        return A_bar_t,B_bar_t,C_t


    def forward(self, x_matrix, seq_length: int = 10):

        outputs = []       
        H_prevt = torch.zeros(self.d_model,self.d_state)
        for i in range(0,seq_length):
            x_t = x_matrix[:,i]

            A_bar_t,B_bar_t,C_t = self.discretize(x_t)

            H_t = (A_bar_t*H_prevt) + (B_bar_t*(x_t.unsqueeze(-1)))
            Y_t = torch.sum(C_t * H_t, dim=-1)
            outputs.append(Y_t)
            H_prevt = H_t

        outputs = torch.stack(outputs, dim=1)
        return outputs, H_t

model = SelectiveSSM(d_model=16, d_state=8)
x_input = torch.randn(16, 10)  # (d_model, seq_len)

y_output, final_state = model(x_input)
print("Output Shape:     ", y_output.shape)     # torch.Size([16, 10])
print("Final State Shape:", final_state.shape)   # torch.Size([16, 8])





