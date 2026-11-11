import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as func
from torch.utils.data import IterableDataset,Dataset, DataLoader, Subset
from rich.live import Live
from rich.console import Console
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import sys
import tiktoken
from datasets import load_dataset
console = Console()
tokenize = tiktoken.encoding_for_model("gpt-2")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# hyperparameters
class config:
    def __init__(self, embs, heads, layers, fdimpemb, batchsize, blocksize):
        # vocab size, total amount of possible i/o tokens
        self.vocab = tokenize.n_vocab
        # embedding dimensions: its kind like how many dimensions the model thinks in? Yeah im not sure either, like why does it need to think in a thousand dimensions???
        self.embs = embs
        # attention heads: its like how many different things the model can think/analyze at once
        self.heads = heads
        # number of layers in the model, simple
        self.layers = layers
        # size of the hidden layers, also simple
        self.ffdim = self.embs * fdimpemb
        # number of chunks being fed at once    
        self.batchsize = batchsize
        # number of tokens in one chunk
        self.blocksize = blocksize
        
@torch.no_grad()
def gen(model, input, maximumamountoftokenssayingitfullyfornoconfusion=1, temp=1, stream=True):
    outputstring = ""
    # tokenize the input to a tensor
    tensorinput = torch.tensor(tokenize.encode(input), dtype=torch.long).to(device)
    # loop for the maximumamountoftokenssayingitfullyfornoconfusion
    for _ in range(maximumamountoftokenssayingitfullyfornoconfusion):
        # run it through model
        modeloutput = model(tensorinput[-model.config.blocksize:].unsqueeze(0).to(device))
        # logits are unconstrained outputs and then you just get the biggest one for greedy but for not then you turn them into percents after normalizing and then apply softmax which turns it in then you do multinomial which actually does the sampling properly
        log = modeloutput[0, -1, :]
        #   greedy decoding, this was the old version which kinda sucked but it would probably be better if i had a computer better than a school laptop at school to train a good model
        if temp == 0:
            indusrivervalley = torch.argmax(log).item()
        # introducing: The all new temperature! Brand new never before done idea ever
        else:
            # normalize the logits by dividing
            log = log / temp
            # softmax converts from logits to actual probabilites
            probs = func.softmax(log,dim=-1)
            # this samples from probabilities
            indusrivervalley = torch.multinomial(probs, num_samples=1).item()
        # end if its <|endoftext|> token
        if indusrivervalley == 50256:
            return outputstring
        # append the token as a string to the outputstring and input
        tensorinput = torch.cat([tensorinput, torch.tensor([indusrivervalley],dtype=torch.long, device=device)], dim=0)
        token = tokenize.decode_single_token_bytes(indusrivervalley).decode('utf-8', errors='replace')
        input += token
        outputstring += token
        if stream:
            print(token, end="", flush=True)
    # return
    if stream:
        print()
    return outputstring
# swiglu activation which is a bit more modern and supposedly more effective
class swiglu(nn.Module):
    def forward(self,x):
        # split inputs into two
        x, gate = x.chunk(2, dim=-1)
        # run one through silu then multiply by the other (who came up with this?)
        return func.silu(gate) * x

class blocklayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        # layernorm one
        self.linearone = nn.LayerNorm(config.embs)
        # calculate attention
        self.attention = selfattention(config)
        # layer norm two
        self.lineartwo = nn.LayerNorm(config.embs)
        # block of layers
        self.block = nn.Sequential(
            # output is double the next input because swiglu halves the input size for outputs amount? Like once again WHO could have POSSIBLY come up with this
            nn.Linear(config.embs,config.ffdim * 2),
            # activation function
            swiglu(),
            # convert from thinking to newly processed embeddings
            nn.Linear(config.ffdim,config.embs)
        )
    def forward(self,x):
        # calculate attention
        x = x + self.attention(self.linearone(x))
        # renormalize and then pass through block
        return x + self.block(self.lineartwo(x))
class selfattention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # makes sure head dim is a whole number (otherwise it causes problems)
        assert config.embs % config.heads == 0
        self.heads = config.heads
        self.embs = config.embs
        # triple the size through a network so it can like learn to understand connection between where the words are and what they mean
        self.att = nn.Linear(config.embs, self.embs * 3)
        self.naoyazenin = nn.Linear(self.embs,self.embs)
        self.naoyazenin.initializethisthingyscaleddifferently = 1
    def forward(self, x):
        # get values so dont have to recalc
        bacthsize, sequencelen, embdim = x.size()
        query, key, value = self.att(x).split(self.embs, dim=2)
        # dimensions of each head
        headdim = embdim // self.heads
        # reshape the tensors so each head can be processed
        query = query.view(bacthsize, sequencelen, self.heads, headdim).transpose(1,2)
        key = key.view(bacthsize, sequencelen, self.heads, headdim).transpose(1,2)
        value = value.view(bacthsize, sequencelen, self.heads, headdim).transpose(1,2)
        # i didnt know this existed so i did the whole thing by myself and it kinda sucked so now im using this
        y = func.scaled_dot_product_attention(query,key,value, is_causal=True)
        # put all the head outputs together again
        y = y.transpose(1, 2).contiguous().view(bacthsize, sequencelen, embdim)
        return self.naoyazenin(y)



class model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # convert vocab inputs into embeddings
        self.embed = nn.Embedding(config.vocab,config.embs)
        # positional embeddings, makes the model able to know the order of tokens
        self.positionalembed = nn.Embedding(config.blocksize, config.embs)
        # hidden layers
        self.onions = nn.ModuleList([
            blocklayer(config) for _ in range(config.layers)
        ])
        # get it its treachery because its the final layer norm and treachery is the final layer like holy dantes inferno/ultrakill ref
        self.treachery = nn.LayerNorm(config.embs)
        # convert from embeddings back into vocab outputs using the same inputs (apparantly works better for some reason? I dont know i just wanted to half parameter count and it did better)
        self.out = lambda x: func.linear(x, self.embed.weight)
        # initialize the weights to a reasonable number so you dont have to spend 3/4 of the training cycle just getting to normal numbers
        self.apply(self.initweight)
    # so the loss was starting at 368 and i googled it you need one of these so the wiehgts dont start too high (stack overflow and thoise little summaries under reddit be carrying [i hate school blocking the only reliable sources of info])
    def initweight(self, module):
        # check if its a Linear lays
        if isinstance(module, nn.Linear):
            standardev = 0.02
           # check if we should initializethisthingyscaleddifferently
            if hasattr(module, 'initializethisthingyscaleddifferently'):
                standardev *= (2 * self.config.layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=standardev)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            
            
        
    def forward(self, x): # x is used in place of inputs (maybe because its like a large math function? y is used for correct outputs too like a math function i think thats it)
        pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
        x = self.embed(x) + self.positionalembed(pos)
        # onion layers
        for onion in self.onions:
            # run through each onion in the bucket
            x = onion(x)
        x = self.treachery(x)
        # crossentropy automatically calculates softmax() which is what you use to convert to probs to sample so we return logits which are like unconstrained values that you apply through softmax to then get probabilities
        return self.out(x) 
class dataset(IterableDataset):
    def __init__(self, blocksize, split="train", shuffbuff=10000):
        super().__init__()
        self.shuffbuff = shuffbuff
        self.split = split
        self.blocksize = blocksize
    def __iter__(self):
        # load the dataset
        stream = load_dataset("HuggingFaceFW/fineweb-edu",name="sample-10BT",split=self.split,streaming=True)
        # shuffle loaded dataset using small 
        stream = stream.shuffle(buffer_size=self.shuffbuff, seed=67)
        # token buffer
        tokens = []
        for larp in stream:
            # get text in larp
            text = larp.get("text","")
            if not text:
                continue
            # extend with tokens then add endoftext
            tokens.extend(tokenize.encode(text))
            tokens.append(50256)
            while len(tokens) >= self.blocksize + 1:
                # niche reference (chunkk)
                chunkk = torch.tensor(tokens[:self.blocksize + 1], dtype=torch.long)
                # return inputs, targets using yield because streaming
                yield chunkk[:-1], chunkk[1:]
                # delete old tokens
                del tokens[:self.blocksize]

# training area
print("Defining loss...")
# loss (very comedy yes yes)
loss = nn.CrossEntropyLoss() or """
  |     |    |  ,
  |     |    |  |
  |     |    |  |
--------|-----------
 |   |  | |
 |   |  | |
 |   |  | | -------
"""
print("Creating config...")
confiigurashen = config(64,4,3,4,4,1024)
# put model on device and set it to training mode(tm)
print("Creating model...")
themodel = model(confiigurashen).to(device).train()
if bool((modelpath := input("Gen?:"))) == True:
    # load model onto device, then put it on device and eval mode
    themodel.load_state_dict(torch.load(modelpath, weights_only=True, map_location=device))
    themodel.to(device).eval()
    try:
        import msvcrt
        s = ""
        print("type and it will show the next token, press enter for full generation")
        while True:
            print("", end ="\x1b[2K")
            print(">>" + s, end="", flush=True)
            # get model output
            texthinggenerate = gen(themodel,(s if s != "" else " ")[-confiigurashen.blocksize:], temp=0, stream=False)
            texthinggenerate = texthinggenerate if texthinggenerate != "\n" else r"\n"
            # print model text in red
            print(f"\033[31m{texthinggenerate}\033[0m", end = "\r")
            try:
                e = msvcrt.getch()
                # check for newlines and carriage return
                if e in (b"\r", b"\n"):
                    print()
                    break
                # backspace
                elif e in (b'\x08', b'\x7f'):
                    if len(s) > 0:
                        s = s[:-1]  
                    continue
                e = e.decode('utf-8')
            except Exception:
                continue
            s += e
    except ImportError:
        s = input("input:")
    nextoken = gen(themodel,s[-confiigurashen.blocksize:], temp=0)
    s += nextoken
    gen(themodel,s,300)
    sys.exit()
print("Defining optimizer...")
# adamW optimizer
op = opt.AdamW(themodel.parameters(), lr=3e-4)
print("Loading data..")
# make the dataset and take the subset of that which is the whole thing but you can change it if you want
parametercount = sum(param.numel() for param in themodel.parameters())
trainsteps = (20*parametercount) // (confiigurashen.blocksize*confiigurashen.batchsize)
larpset = dataset(blocksize=confiigurashen.blocksize)
traindataload = DataLoader(larpset,batch_size=confiigurashen.batchsize,num_workers=0,drop_last=True)
print("Scheduling learning...")
schedule = CosineAnnealingLR(op,T_max=trainsteps,eta_min=3e-5)
# find parameter count and print as well as the number of steps

step = 0
print(f"Training {parametercount} parameters with {trainsteps} steps on {device}:")
try:
    # epochs
    for epoch in range(1):
        # the trainbatch from load
        for inputs, targets in traindataload:
            # end if pass training step numbers
            if step > trainsteps:
                break
            op.zero_grad()
            # get inputs and targets on device
            inp = inputs.to(device)
            targ = targets.to(device)
            # calculate logits and loss (with bf16 if supported on device)
            if torch.cuda.is_bf16_supported():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = themodel(inp)
                    B, T, C = logits.shape
                    lossv = loss(logits.view(B * T, C), targ.view(B * T))
            else:
                logits = themodel(inp)
                B, T, C = logits.shape
                lossv = loss(logits.view(B * T, C), targ.view(B * T))
            # backward pass
            lossv.backward()
            # step optimizer and LR schedule
            op.step()
            schedule.step()
            # increase step
            step += 1
            # print out info
            if step % 100 == 0:
                print(f'loss: {lossv.item()}, step: {step}, lr = {schedule.get_last_lr()}')
                if step % 1000 == 0:
                    print('the', end='', flush=True)
                    gen(themodel, 'the', 100)
except (KeyboardInterrupt, Exception) as e:
    print(e)
    pass
# set to eval for saving
themodel.eval()
# save model state_dict
if bool(savepath := input("Save?:")) == True:
    torch.save(themodel.state_dict(),savepath)