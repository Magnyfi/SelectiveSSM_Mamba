import torch 
import torch.nn as nn

#creating synthetic data
# take a series of L tokens out of which K are content tokens and the rest are noise
# noise is denoted by number 0 and content tokens are denoted by numbers from 1 to L

num_content_token = 30
vocab_size = 32
d_model = 16

def synthetic_data(batch,L:int = 64,K:int = 16):
    
    def generate(L,K):
        num_content_token = 30
        noise_token = 0
        delim_token = num_content_token + 1

        positions = torch.randperm(L)[:K] #chooses K indexes in a row of length L
        positions,_ = torch.sort(positions)
        content_tokens = torch.randint(1,num_content_token+1,(K,))

        data = torch.full((L,),noise_token)
        data[positions] = content_tokens

        delim_ext = torch.full((K+1,),noise_token)
        delim_ext[0] = delim_token

        data_final = torch.cat((data,delim_ext),dim = 0) 

        target = content_tokens                            

        mask = torch.zeros_like(data_final, dtype=torch.bool)
        mask[-K:] = True     

        return data_final, target, mask

    seq_length = L+K+1
    seqs = []
    targets = []
    masks = []
    for i in range(batch):
        seq,target,mask = generate(L,K)
        seqs.append(seq)
        targets.append(target)
        masks.append(mask)

    seqs = torch.stack(seqs) 
    targets = torch.stack(targets)
    masks = torch.stack(masks)

   
    return seqs,targets,masks

seqs,targets,masks = synthetic_data(batch = 3)

