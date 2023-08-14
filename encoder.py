from all_imports import *
from multi_head_attention import MultiheadAttention
from scoring import compute_GAD_scores
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.nonlinear_net = nn.Sequential(    #non_linear_net
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,batch_idx,mode,GAD_scores,mask):
        # Attention part
        #GAD_scores={}
        attn_out = self.self_attn(x,mask=mask)
        x = x + self.dropout(attn_out)  #residual  connection
        x = self.norm1(x)               #layer norm
        #print(attn_out.shape,attn_out)
        #print('COmputing gad scores::::::\n')
        if mode=='test':
            GAD_scores=compute_GAD_scores(attn_out,GAD_scores,batch_idx)
        else:
            GAD_scores={}
        
        #sys.exit(0)
        # MLP part
        nonlinear_out = self.nonlinear_net(x)
        x = x + self.dropout(nonlinear_out)
        x = self.norm2(x)
        

        return x,attn_out,GAD_scores

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x,batch_idx,mode,GAD_scores={}, mask=None):
        for layer in self.layers:
            x,y,GAD_scores = layer(x,batch_idx,mode,GAD_scores, mask)
        #print(x,x.size(),'in encoder...............')
        return x,y,GAD_scores

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps