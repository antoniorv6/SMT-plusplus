import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):

    def __init__(self, embedding_dim, num_heads=None, dropout=0, proj_value=True, use_flash_attn=True) -> None:
        super().__init__()

        self.proj_value = proj_value
        self.lq = nn.Linear(embedding_dim, embedding_dim)
        self.lk = nn.Linear(embedding_dim, embedding_dim)
        if proj_value:
            self.lv = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.use_flash_attention = use_flash_attn
            
    def forward(self, query, key, value, key_pad_mask=None, attn_mask=None, get_weights=True):
        
        target_len, b, c = query.size()
        source_len = key.size(0)

        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value) if self.proj_value else value

        q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        
        if self.use_flash_attention:
            q = q.reshape(-1, target_len, self.head_dim)
            k = k.reshape(-1, source_len, self.head_dim)
            v = v.reshape(-1, source_len, self.head_dim)
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                attn_output, attn_output_weights = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
                attn_output = torch.reshape(attn_output, (target_len, b, self.num_heads*self.head_dim))
                attn_output = attn_output.reshape(b, self.num_heads, target_len, self.head_dim).transpose(1, 2).contiguous()
                attn_output = attn_output.view(target_len, b, c)
                attn_output = self.out_proj(attn_output)
                return attn_output, None
        
        else:
            attn_weights = torch.bmm(q, k.transpose(1,2))

            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
                if attn_mask.dtype == torch.bool:
                    attn_weights.masked_fill_(attn_mask, float("-inf"))
                else:
                    attn_weights += attn_mask

            if key_pad_mask is not None:
                attn_output_weigths = attn_weights.view(b, self.num_heads, target_len, source_len)
                attn_output_weigths = attn_weights.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
                attn_output_weigths = attn_weights.view(b*self.num_heads, target_len, source_len)

            attn_output_weigths_raw = self.softmax(attn_output_weigths)
            attn_output_weigths = self.dropout(attn_output_weigths_raw)
            attn_output = torch.bmm(attn_output_weigths, v)
            attn_output = attn_output.transpose(0,1).contiguous().view(target_len, b, c)
            attn_output = self.out_proj(attn_output)

            if get_weights:
                attn_output_weigths_raw = attn_output_weigths_raw.view(b, self.num_heads, target_len, source_len)
                return attn_output, attn_output_weigths_raw.sum(dim=1) / self.num_heads

            return attn_output

query = torch.rand(1024, 1, 256, device="cuda")
key = torch.rand(1024, 1, 256, device="cuda")
value = torch.rand(1024, 1, 256, device="cuda")

mha = MHA(256, 4, 0).to(torch.device("cuda"))

z = mha(query, key, value)
print(z.size())


#with torch.backends.cuda.sdp_kernel(enable_math=False):
#    z = F.scaled_dot_product_attention(query,key,value)