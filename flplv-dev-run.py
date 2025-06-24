from backend.data_processing import MnistLoader
from backend.data_processing import SquareImageSplitingLoader
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import wandb
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super().__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
        
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class ImageEncoder(nn.Module):
    
    def __init__(
            self, d_img_patch, d_input, n_layers, n_head, d_k, d_v, d_inner, dropout=0.1, n_position=200):

        super().__init__()

        self.compressor = nn.Linear(d_img_patch, d_input)
        self.position_enc = PositionalEncoding(d_input, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_input, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, img_embedding, src_mask=None, return_attns=False):

        enc_slf_attn_list = []

        enc_output = self.dropout(self.position_enc(self.compressor(img_embedding)))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
    

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))
    
class DigitClassifier(nn.Module):
    def __init__(self, d_input):
        super().__init__()
        self.classifier = nn.Linear(d_input, 10)
        
    def forward(self, x):
        x = self.classifier(x)
        return x

def dev_run():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class Hyperparameters():
        def __init__(self):
            self.d_img = 28 * 28 # dimension of the input image
            self.d_img_patch = 7 * 7   # dimension of each patch
            self.n_img_patches = (self.d_img // self.d_img_patch)
            self.d_input = 10  # dimensions of compressed embeddings
            self.bs = 1024 * 10  # batch size for training
            self.n_layers = 8  # number of encoder layers in the transformer
            self.n_heads = 3
            self.d_model = 64
            self.d_hid = 256  # hidden dimension in the feed-forward network
            self.dropout = 0  # dropout rate for the transformer layers
    hp = Hyperparameters()
    
    mnist_loader = MnistLoader(batch_size = hp.bs)
    train_loader, validation = mnist_loader.get_loaders()
    train_loader = SquareImageSplitingLoader(train_loader, hp.n_img_patches, int(hp.d_img_patch ** 0.5))
    validation = SquareImageSplitingLoader(validation, hp.n_img_patches, int(hp.d_img_patch ** 0.5))
    
    encoder = ImageEncoder(
        d_img_patch=hp.d_img_patch,
        d_input=hp.d_input,
        n_layers=hp.n_layers,
        n_head=hp.n_heads,
        d_k=hp.d_model,
        d_v=hp.d_model,
        d_inner=hp.d_hid,
        dropout=hp.dropout,
        n_position=hp.n_img_patches)
    encoder.to(dev)
    
    digits = DigitClassifier(d_input=hp.d_input)
    digits.to(dev)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
                    {'params': encoder.parameters(), 'lr': 0.001},
                    {'params': digits.parameters(), 'lr': 0.001},
                ])
    
    wandb.init(entity="flplv-private", project='mli-week-03-encoder-training')
    
    wandb_step = 0
    for epoch in range(1000):
        epoch_loader = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        
        for batch_idx, (data, labels) in enumerate(epoch_loader):
            data = data.reshape(-1, hp.n_img_patches, hp.d_img_patch)
            data = data.to(dev)
            labels = labels.to(dev)
            
            logits = digits(encoder(data)[0].mean(dim=1))  # mean pooling over patches
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            wandb.log({'loss': loss.item()}, step=wandb_step)
            wandb_step += len(labels)
            
        # epoch end validation
        def accuracy(outputs, labels, dev):
            _, preds = torch.max(outputs, dim = 1)
            return(torch.tensor(torch.sum(preds == labels).item()/ len(preds), device=dev))
        
        accs = []
        for (data, labels) in validation:
            data = data.reshape(-1, hp.n_img_patches, hp.d_img_patch)
            data = data.to(dev)
            labels = labels.to(dev)
            logits = digits(encoder(data)[0].mean(dim=1))
            acc = accuracy(logits, labels, dev=dev)
            accs.append(acc)
        
        epoch_acc = torch.stack(accs).mean()
        wandb.log({'acc': epoch_acc.item()}, step=wandb_step)
            
    wandb.finish()
        
if __name__ == "__main__":
    dev_run()