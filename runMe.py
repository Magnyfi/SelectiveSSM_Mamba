import torch
from SSmodel import SelectiveSSM,SSM_withoutExp
from matplotlib import pyplot as plt
import numpy as np

x_input = torch.randn(1,2500, 16)  # (batch,seq_length, d_model)

model_exp = SelectiveSSM(d_model=16, d_state=8)
model_noExp = SSM_withoutExp(d_model=16, d_state=8)

y_exp, H_exp = model_exp(x_input)
y_noExp, H_noExp = model_noExp(x_input)
print(H_exp.shape)

def process(H):
    print(H.shape)
    H = H.detach()[0]
    print(H.shape)                  # (seq_len, d_model, d_state)
    H_flat = H.reshape(H.shape[0], -1)
    print(H_flat.shape)     # (seq_len, d_model*d_state)
    return H_flat.norm(dim=-1).cpu().numpy()

H_exp = process(H_exp)
H_noExp = process(H_noExp)


def first_inf_index(arr):
    inf_idx = np.where(np.isinf(arr))[0]
    return inf_idx[0] if len(inf_idx) > 0 else len(arr)

cutoff = min(first_inf_index(H_exp), first_inf_index(H_noExp))
print(f"Truncating plot at t={cutoff} (first NaN encountered)")

t = range(cutoff)

print(H_noExp[:50])


plt.plot(t, H_noExp[:cutoff], label="Without exp", linewidth=1.5)
plt.plot(t, H_exp[:cutoff], label="With exp (stabilized)", linewidth=1.5)
plt.yscale("log")
plt.xlabel("Timestep t")
plt.ylabel("‖H_t‖ (hidden state norm)")
plt.title("Hidden State Magnitude Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#print("Output Shape:     ", y_output.shape)     # torch.Size([4,10, 16])
#print("Final State Shape:", final_state.shape)   # torch.Size([4,16, 8])
