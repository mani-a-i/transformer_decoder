import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters:
batch_size = 64 # how many independent sequence we will process in parallel
block_size = 256  # what is the maximum context length for prediction
max_iter = 5001
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


with open('input.txt','r',encoding='utf-8') as f:
  text = f.read()

# All unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)    # len of vocab i.e, number if unique characters

# creating a mapping from ch to integers and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
ios = {i:ch for i,ch in enumerate(chars)}

# encode a string to list of integers. decode a list of integers to string
encode = lambda s : [stoi[ch] for ch in s]
decode = lambda ls_i : ''.join([ios[i] for i in ls_i])

# encode the entire text
data = torch.tensor(encode(text),dtype = torch.long)

# create train and val split
n = int(0.9*len(data))      # 90% of data as train
train_data = data[:n]
val_data = data[n:]

# Data Loading
def get_batch(split):

    # Generate a small batch of data with input x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(high = len(data)-block_size, size=(batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)

    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# head_size is the size of indiviuaal head and not of the multihead.

class Head(nn.Module):
#    """ One head of self attention """
    #  k,q and v vector -----> (B,T,head_size)

    def __init__(self,head_size) :
       super().__init__()
       self.key = nn.Linear(n_embed,head_size,bias = False)
       self.query = nn.Linear(n_embed,head_size,bias = False)
       self.value = nn.Linear(n_embed,head_size,bias = False)
       self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
       self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        
        B,T,C = x.shape        

        k = self.key(x)      # (B,T,head_size)
        q = self.query(x)    # (B,T,head_size)
        v = self.value(x)    # (B,T,head_size)

        # Compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * C**-0.5     # -----> (B,T,T)

        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))    # ---> (B,T,T)
        wei = F.softmax(wei,dim=-1)
        wei  = self.dropout(wei)
        # Perform the weighted aggregation of values.
        out = wei @ v      # -----> (B,T,head_size)        

        return out    


class MultiHeadAttention(nn.Module):
   
    def __init__(self,num_heads,head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):        
        out =  torch.cat([h(x) for h in self.heads],dim=-1) 
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self,n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed*4),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(dropout)
        )

    def forward(self,x):        
        return self.net(x)

class Block(nn.Module):

    def __init__(self,n_embed,num_heads)    -> None:
        super().__init__()

        head_size = n_embed//num_heads
        self.sa = MultiHeadAttention(num_heads,head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)  
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        # LayerNorm happens before multihead attention and FFL
        x = x + self.sa(self.ln1(x))    
        x = x + self.ffwd(self.ln2(x))
        return x         


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # Positional Embedding
        self.positional_embedding_tabel = nn.Embedding(block_size,n_embed)

        # Blocks    
        self.blocks = nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layer)])

        # final layernorm
        self.lnf = nn.LayerNorm(n_embed)

        # language model head. 
        self.lm_head = nn.Linear(n_embed,vocab_size) # gives embedding    
 
    def forward(self, idx, targets=None):
        
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embed)
        pos_emb = self.positional_embedding_tabel(torch.arange(T,device=device))    # eg: (T,n_embed)

        x = tok_emb + pos_emb         # (B,T,n_embed)

        # Multi head Self attention  layer()
        x = self.blocks(x)           # (B,T,n_embed) 

        x = self.lnf(x)      # final layernorm             
        
        # linear layer
        logits = self.lm_head(x)    # (B,T,vocab_size)


        # if taget is present find loss else return only logits
        if targets != None:     
          B,T,C = logits.shape

          logits = logits.view(B*T,C)     # -----> (B*T,C)  eg:(32,64)
          targets = targets.view(B*T)     # -----> (B*T)  eg:(32)
          loss = F.cross_entropy(logits,targets)    # softmax(logits, dim=1) + negative log likelyhood
          return logits,loss

        else:
          return logits

    def generate(self,idx,max_tokens): # how many tokens to generate

        for _ in range(max_tokens):

          # if idx has more than 8 tokens. Then take last 8 tokens
          idx_cond = idx[:,-8:]

          # logits of starting token 
          logits = self(idx_cond) 

          # find the last token from every batch     
          logits = logits[:,-1,:]        

          # prob of each token       
          probs = F.softmax(logits,dim = -1)    
          idx_next = torch.multinomial(probs,num_samples = 1,)   
          idx = torch.cat((idx,idx_next),dim=1)

        # print(idx.shape)
        return idx    

# Creating a model.
model = BigramLanguageModel()
# move model parameters to device. In our case its the embedding table
m = model.to(device)  


opt = torch.optim.AdamW(m.parameters(),lr=learning_rate) 


for iter in range(max_iter):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


# Moving the starting token to device.
context = torch.zeros((1,1),dtype = torch.long,device=device)

print(decode(m.generate(context,max_tokens=500)[0].tolist()))














