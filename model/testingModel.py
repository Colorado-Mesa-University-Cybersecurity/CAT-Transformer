import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
    
#All layers of the model
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys =nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)


    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) #(batch_size, head_dim, #query_embeddings, attention maps)

        #If FT, then query_embeddings = # attention maps
        if(attention.shape[-1] == attention.shape[-2]):
            #We only want the CLS Token's attention maps
            energy_temp = energy[:,:,0,1:key_len].unsqueeze(2)
            attention_temp = torch.softmax(energy_temp / (self.embed_size ** (1/2)), dim=3)
            # print("cls attention", cls_attention.shape)
            avg_attention = attention_temp.mean(0)
            # print("cls attention", avg_attention.shape)
            avg_attention = avg_attention.mean(0)
            # print("cls attention", avg_attention.shape)

        else:
            # Calculate simplified attention scores
            avg_attention = attention.mean(0)  # Average across batches
            # print("batch average", avg_attention.shape)
            avg_attention = avg_attention.mean(0).squeeze(dim=0)
            # print("head average", avg_attention.shape)


        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim) #(batch_size, n_features, embed_size)
        out = self.fc_out(out)

        return out, avg_attention
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, pre_norm_on):
        super(TransformerBlock, self).__init__()

        self.pre_norm_on = pre_norm_on
        if self.pre_norm_on:
            self.pre_norm = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion*embed_size),
                                          nn.GELU(),
                                          nn.Linear(forward_expansion*embed_size, embed_size)
                                          )
        self.dropout = nn.Dropout(dropout)

    def forward(self,value,key,query):
        if self.pre_norm_on:
            query = self.pre_norm(query)
            key = self.pre_norm(key)
            value = self.pre_norm(value)
            
        attention, avg_attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, avg_attention
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, pre_norm_on):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion, pre_norm_on)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key):
        out, avg_attention = self.transformer_block(value, key, x)

        return out, avg_attention

class Decoder(nn.Module):
    def __init__(self,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 decoder_dropout,
                 pre_norm_on,
                 FT_on,
                 get_attn
    ):
        super(Decoder, self).__init__()

        self.FT_on = FT_on

        self.get_attn = get_attn

        self.layers = nn.ModuleList(
                [
                    DecoderBlock(
                        embed_size,
                        heads,
                        dropout=decoder_dropout,
                        forward_expansion=forward_expansion,
                        pre_norm_on=pre_norm_on
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, class_embed, context):
        attention_scores = []
        for layer in self.layers:
            # x is the classification embedding (CLS Token)
            # context are the feature embeddings that will be used as key and value
            x, attention = layer(class_embed, context, context)
            attention_scores.append(attention)
 
            class_embed = x #output of previous layer acts as input to next layer
            if self.FT_on:
                context = class_embed
        
        #for the last output, only the contextualized CLS token is passed for FT
        #with CAT, the contextualized CLS token is already the only output
        if self.FT_on:
            x = x[:,0] #Extracting out the CLS token 
        if self.get_attn:
            return x, torch.stack(attention_scores)
        return x

class lin(nn.Module):
    def __init__(self, sigma, embed_size, n_cont, cat_feat, num_target_labels):
        super(lin, self).__init__()

        self.sigma = sigma
        self.embed_size = embed_size
        self.n_cont = n_cont
        self.cat_feat_on = False
        if len(cat_feat)!=0:
            self.cat_feat_on=True

        self.lin_embed = nn.ModuleList([nn.Linear(in_features=1, out_features=self.embed_size) for _ in range(n_cont)]) # each feature gets its own linear layer

        if self.cat_feat_on:
            self.cat_embeddings = nn.ModuleList([nn.Embedding(num_classes, embed_size) for num_classes in cat_feat])
            
        #CLS Token
        self.target_label_embed = nn.ModuleList([nn.Embedding(1, self.embed_size) for _ in range(num_target_labels)])

    def forward(self, x_cat, x_cont):
        x = x_cont.unsqueeze(2) #(batch_size, n_features) -> (batch_size, n_features, 1)
        embeddings = []
        for i, e in enumerate(self.lin_embed):
            goin_in = x[:,i,:]
            goin_out = e(goin_in)
            embeddings.append(goin_out)

        if self.cat_feat_on:
            cat_x = x_cat.unsqueeze(2)
            for i, e in enumerate(self.cat_embeddings):
                goin_in = cat_x[:,i,:]
                goin_out = e(goin_in)
                goin_out=goin_out.squeeze(1)
                embeddings.append(goin_out)

        target_label_embeddings_ = []
        for e in self.target_label_embed:
            input = torch.tensor([0], device=x.device)
            temp = e(input)
            temp = temp.repeat(x.size(0), 1)
            target_label_embeddings_.append(temp)

        class_embeddings = torch.stack(target_label_embeddings_, dim=1)

        context = torch.stack(embeddings, dim=1)

        return class_embeddings, context

class ConstantPL(nn.Module):
    def __init__(self, sigma, embed_size, n_cont, cat_feat, num_target_labels):
        super(ConstantPL, self).__init__()

        self.sigma = sigma
        self.embed_size = embed_size
        self.n_cont = n_cont
        self.cat_feat_on = False
        if len(cat_feat)!=0:
            self.cat_feat_on=True

        coefficients = torch.normal(0, self.sigma, (n_cont, self.embed_size//2))

        self.register_buffer('coefficients', coefficients)

        self.lin_embed = nn.ModuleList([nn.Linear(in_features=self.embed_size, out_features=self.embed_size) for _ in range(n_cont)]) # each feature gets its own linear layer

        if self.cat_feat_on:
            self.cat_embeddings = nn.ModuleList([nn.Embedding(num_classes, embed_size) for num_classes in cat_feat])
            
        #CLS Token
        self.target_label_embed = nn.ModuleList([nn.Embedding(1, self.embed_size) for _ in range(num_target_labels)])

    def forward(self, x_cat, x_cont):
        x = x_cont.unsqueeze(2) #(batch_size, n_features) -> (batch_size, n_features, 1)

        temp = []
        for i in range(self.n_cont):
            input = x[:,i,:]
            out = torch.cat([torch.cos(self.coefficients[i,:] * input), torch.sin(self.coefficients[i,:] * input)], dim=-1)
            temp.append(out)
        
        embeddings = []
        x = torch.stack(temp, dim=1)
        for i, e in enumerate(self.lin_embed):
            goin_in = x[:,i,:]
            goin_out = e(goin_in)
            embeddings.append(goin_out)

        if self.cat_feat_on:
            cat_x = x_cat.unsqueeze(2)
            for i, e in enumerate(self.cat_embeddings):
                goin_in = cat_x[:,i,:]
                goin_out = e(goin_in)
                goin_out=goin_out.squeeze(1)
                embeddings.append(goin_out)

        target_label_embeddings_ = []
        for e in self.target_label_embed:
            input = torch.tensor([0], device=x.device)
            temp = e(input)
            temp = temp.repeat(x.size(0), 1)
            target_label_embeddings_.append(temp)

        class_embeddings = torch.stack(target_label_embeddings_, dim=1)

        context = torch.stack(embeddings, dim=1)

        return class_embeddings, context

class PL(nn.Module):
    def __init__(self, sigma, embed_size, n_cont, cat_feat, num_target_labels):
        super(PL, self).__init__()

        self.sigma = sigma
        self.embed_size = embed_size
        self.n_cont = n_cont
        self.cat_feat_on = False
        if len(cat_feat)!=0:
            self.cat_feat_on=True

        coefficients = torch.normal(0, self.sigma, (n_cont, self.embed_size//2))

        self.coefficients = nn.Parameter(coefficients)

        self.lin_embed = nn.ModuleList([nn.Linear(in_features=self.embed_size, out_features=self.embed_size) for _ in range(n_cont)]) # each feature gets its own linear layer

        if self.cat_feat_on:
            self.cat_embeddings = nn.ModuleList([nn.Embedding(num_classes, embed_size) for num_classes in cat_feat])
            
        #CLS Token
        self.target_label_embed = nn.ModuleList([nn.Embedding(1, self.embed_size) for _ in range(num_target_labels)])

    def forward(self, x_cat, x_cont):
        x = x_cont.unsqueeze(2) #(batch_size, n_features) -> (batch_size, n_features, 1)

        temp = []
        for i in range(self.n_cont):
            input = x[:,i,:]
            out = torch.cat([torch.cos(self.coefficients[i,:] * input), torch.sin(self.coefficients[i,:] * input)], dim=-1)
            temp.append(out)
        
        embeddings = []
        x = torch.stack(temp, dim=1)
        for i, e in enumerate(self.lin_embed):
            goin_in = x[:,i,:]
            goin_out = e(goin_in)
            embeddings.append(goin_out)

        if self.cat_feat_on:
            cat_x = x_cat.unsqueeze(2)
            for i, e in enumerate(self.cat_embeddings):
                goin_in = cat_x[:,i,:]
                goin_out = e(goin_in)
                goin_out=goin_out.squeeze(1)
                embeddings.append(goin_out)

        target_label_embeddings_ = []
        for e in self.target_label_embed:
            input = torch.tensor([0], device=x.device)
            temp = e(input)
            temp = temp.repeat(x.size(0), 1)
            target_label_embeddings_.append(temp)

        class_embeddings = torch.stack(target_label_embeddings_, dim=1)

        context = torch.stack(embeddings, dim=1)

        return class_embeddings, context

class ExpFF(nn.Module):
    def __init__(self, alpha, embed_size, n_cont, cat_feat, num_target_labels):
        super(ExpFF, self).__init__()

        self.alpha = alpha
        self.embed_size = embed_size
        self.n_cont = n_cont
        self.cat_feat_on = False
        if len(cat_feat)==0:
            self.cat_feat_on=True

        coefficients = self.alpha ** (torch.arange(self.embed_size//2) / self.embed_size//2) #Each feature shares the same set of scaling factors
        coefficients = coefficients.unsqueeze(0)

        self.register_buffer('embedding_coefficients', coefficients)

        self.lin_embed = nn.ModuleList([nn.Linear(in_features=self.embed_size, out_features=self.embed_size) for _ in range(n_cont)]) # each feature gets its own linear layer

        if self.cat_feat_on:
            self.cat_embeddings = nn.ModuleList([nn.Embedding(num_classes, embed_size) for num_classes in cat_feat])
            
        #CLS Token
        self.target_label_embed = nn.ModuleList([nn.Embedding(1, self.embed_size) for _ in range(num_target_labels)])

    def forward(self, x_cat, x_cont):
        x = x_cont.unsqueeze(2) #(batch_size, n_features) -> (batch_size, n_features, 1)

        temp = []
        for i in range(self.n_cont):
            input = x[:,i,:]
            #(1,80)x(256,1)
            out = torch.cat([torch.cos(self.embedding_coefficients * input), torch.sin(self.embedding_coefficients * input)], dim=-1)
            temp.append(out)
        
        embeddings = []
        x = torch.stack(temp, dim=1)
        for i, e in enumerate(self.lin_embed):
            goin_in = x[:,i,:]
            goin_out = e(goin_in)
            embeddings.append(goin_out)

        if self.cat_feat_on:
            cat_x = x_cat.unsqueeze(2)
            for i, e in enumerate(self.cat_embeddings):
                goin_in = cat_x[:,i,:]
                goin_out = e(goin_in)
                goin_out=goin_out.squeeze(1)
                embeddings.append(goin_out)

        target_label_embeddings_ = []
        for e in self.target_label_embed:
            input = torch.tensor([0], device=x.device)
            temp = e(input)
            temp = temp.repeat(x.size(0), 1)
            target_label_embeddings_.append(temp)

        class_embeddings = torch.stack(target_label_embeddings_, dim=1)

        context = torch.stack(embeddings, dim=1)

        return class_embeddings, context
    

class ClassificationHead(nn.Module):
    def __init__(self, embed_size, dropout, mlp_scale_classification, num_target_classes):
        super(ClassificationHead, self).__init__()
        
        #flattening the embeddings out
        self.input = embed_size
        self.lin1 = nn.Linear(self.input, mlp_scale_classification*self.input)
        self.norm = nn.LayerNorm(self.input)

        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(mlp_scale_classification*self.input, num_target_classes)
        # self.lin3 = nn.Linear(mlp_scale_classification*self.input, self.input)
        # self.lin4 = nn.Linear(self.input, num_target_classes)
        self.relu = nn.GELU()

        self.initialize_weights()

    def initialize_weights(self): #he_initialization.
        torch.nn.init.kaiming_normal_(self.lin1.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.lin1.bias)

        torch.nn.init.kaiming_normal_(self.lin2.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.lin2.bias)

    def forward(self, x):

        x= torch.reshape(x, (-1, self.input))

        x = self.norm(x)
        x = self.relu(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        # x = self.lin1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.lin2(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.lin3(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.lin4(x)
  
        return x

class RegressionHead(nn.Module):
    def __init__(self, embed_size, dropout, mlp_scale_classification):
        super(RegressionHead, self).__init__()
        
        #flattening the embeddings out so each sample in batch is represented with a 460 dimensional vector
        self.input = embed_size
        self.lin1 = nn.Linear(self.input, mlp_scale_classification*self.input)
        self.norm = nn.LayerNorm(self.input)
        self.lin2 = nn.Linear(mlp_scale_classification*self.input, 1)
        self.drop = nn.Dropout(dropout)
        # self.lin2 = nn.Linear(mlp_scale_classification*self.input, mlp_scale_classification*self.input)
        # self.lin3 = nn.Linear(mlp_scale_classification*self.input, 1)
        # self.lin4 = nn.Linear(self.input, 1)
        self.relu = nn.GELU()


        self.initialize_weights()

    def initialize_weights(self): #he_initialization.
        torch.nn.init.kaiming_normal_(self.lin1.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.lin1.bias)

        torch.nn.init.kaiming_normal_(self.lin2.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.lin2.bias)

    def forward(self, x):

        x= torch.reshape(x, (-1, self.input))

        x = self.norm(x)
        x = self.relu(x)
        x = self.lin1(x)
        x= self.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        # x = self.lin1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.lin2(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.lin3(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.lin4(x)
  
        return x

class MyFTTransformer(nn.Module):
    def __init__(self, 
                 embedding = 'ConstantPL',
                 alpha=0.5, # Used to initialize the coefficients for the Exponential FF 
                 embed_size=200,
                 n_cont = 0,
                 cat_feat:list = [], # ex: [10,4] - 10 categories in the first column, 4 categories in the second column
                 num_layers=1, #Transformer layers
                 heads=10, 
                 forward_expansion=8, # Determines how wide the Linear Layers are the encoder. Its a scaling factor. 
                 decoder_dropout=0.1,
                 classification_dropout = 0.2,
                 pre_norm_on = False,
                 mlp_scale_classification = 8, #Scaling factor for linear layers in head
                 regression_on = False,
                 targets_classes : list=  [3],
                 get_attn = False
                 ):
        super(MyFTTransformer, self).__init__()

        self.regression_on = regression_on

        self.get_attn = get_attn

        if embedding == 'Exp':
            self.embeddings = ExpFF(alpha=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
        elif embedding == 'PL':
            self.embeddings = PL(sigma=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
        elif embedding == 'ConstantPL':
            self.embeddings = ConstantPL(sigma=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
        else:
            self.embeddings = lin(sigma=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
            
        self.decoder = Decoder(embed_size=embed_size, num_layers=num_layers, heads=heads, forward_expansion=forward_expansion, 
                               decoder_dropout=decoder_dropout, pre_norm_on=pre_norm_on, FT_on=True, get_attn=self.get_attn)
        if not regression_on:
            self.out_head = ClassificationHead(embed_size=embed_size, dropout=classification_dropout, 
                                                                   mlp_scale_classification=mlp_scale_classification, 
                                                                   num_target_classes=targets_classes[0])
        else:
            self.out_head = RegressionHead(embed_size=embed_size, dropout=classification_dropout, mlp_scale_classification=mlp_scale_classification)

    def forward(self, cat_x, cont_x):
        class_embed, context = self.embeddings(cat_x, cont_x)

        x = torch.cat([ class_embed, context], dim = 1) #Concatenate CLS and context together

        if self.get_attn:
            x, avg_attention = self.decoder(x, x) # Keys Queries and Values all work with the concatenated tokens
        else:
            x = self.decoder(x,x)

        output = self.out_head(x)

        if self.get_attn:
            return output, avg_attention
        else:
            return output

class CATTransformer(nn.Module):
    def __init__(self, 
                 embedding = 'ConstantPL',
                 alpha=0.5, # Used to initialize the coefficients for the Exponential FF 
                 embed_size=200,
                 n_cont = 0,
                 cat_feat:list = [], # ex: [10,4] - 10 categories in the first column, 4 categories in the second column
                 num_layers=1, #Transformer layers
                 heads=10, 
                 forward_expansion=8, # Determines how wide the Linear Layers are the transformer. Its a scaling factor. 
                 decoder_dropout=0.1,
                 classification_dropout = 0.2,
                 pre_norm_on = False,
                 mlp_scale_classification = 8, #Scaling factor for linear layers in head
                 regression_on = False,
                 targets_classes : list=  [3],
                 get_attn = False
                 ):
        super(CATTransformer, self).__init__()

        self.regression_on = regression_on

        self.get_attn = get_attn

        if embedding == 'Exp':
            self.embeddings = ExpFF(alpha=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
        elif embedding == 'PL':
            self.embeddings = PL(sigma=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
        elif embedding == 'ConstantPL':
            self.embeddings = ConstantPL(sigma=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
        else:
            self.embeddings = lin(sigma=alpha, embed_size=embed_size, n_cont=n_cont, cat_feat=cat_feat,
                                num_target_labels=len(targets_classes))
            
        self.decoder = Decoder(embed_size=embed_size, num_layers=num_layers, heads=heads, forward_expansion=forward_expansion, 
                               decoder_dropout=decoder_dropout, pre_norm_on=pre_norm_on, FT_on=False, get_attn=self.get_attn)
        if not regression_on:
            self.out_head = ClassificationHead(embed_size=embed_size, dropout=classification_dropout, 
                                                                   mlp_scale_classification=mlp_scale_classification, 
                                                                   num_target_classes=targets_classes[0])
        else:
            self.out_head = RegressionHead(embed_size=embed_size, dropout=classification_dropout, mlp_scale_classification=mlp_scale_classification)

    def forward(self, cat_x, cont_x):
        class_embed, context = self.embeddings(cat_x, cont_x)

        if self.get_attn:
            x, avg_attention = self.decoder(class_embed, context)
        else:
            x = self.decoder(class_embed, context)
        
        # for i, e in enumerate(self.heads):
        #     input = x[:, i,:]
        #     output = e(input)
           
        output = self.out_head(x)

        if self.get_attn:
            return output, avg_attention
        else:
            return output

# Dataset loaders for different cases
class Cont_Dataset(Dataset):
    def __init__(self, df : pd.DataFrame, num_columns,task1_column):
        self.n = df.shape[0]
        
        self.task1_labels = df[task1_column].astype(np.int64).values

        self.num = df[num_columns].astype(np.float32).values


    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Retrieve features and labels from the dataframe using column names
        num_features = self.num[idx]
        labels_task1 = self.task1_labels[idx]

        return num_features, labels_task1
    
class Cat_Cont_Dataset(Dataset):
    def __init__(self, df : pd.DataFrame, cat_columns, num_columns,task1_column):
        self.n = df.shape[0]
        
        self.task1_labels = df[task1_column].astype(np.float32).values

        self.cate = df[cat_columns].astype(np.int64).values
        self.num = df[num_columns].astype(np.float32).values


    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Retrieve features and labels from the dataframe using column names
        cat_features = self.cate[idx]
        num_features = self.num[idx]
        labels_task1 = self.task1_labels[idx]

        return cat_features, num_features, labels_task1
    
class Combined_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, cat_columns, num_columns, task1_column):
        self.n = df.shape[0]

        self.task1_labels = df[task1_column].astype(np.float32).values

        # If categorical columns exist, load them; otherwise, initialize as an empty tensor
        if len(cat_columns) > 0:
            self.cate = df[cat_columns].astype(np.int64).values
        else:
            self.cate = torch.empty((self.n, 0))  # Empty tensor when no categorical features

        self.num = df[num_columns].astype(np.float32).values

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Retrieve features and labels from the dataframe using column names
        cat_features = self.cate[idx]
        num_features = self.num[idx]
        labels_task1 = self.task1_labels[idx]

        return cat_features, num_features, labels_task1
    
def rmse(y_true, y_pred):
    # Calculate the squared differences
    squared_diff = (y_true - y_pred)**2

    try:
        # Calculate the mean of the squared differences
        mean_squared_diff = torch.mean(squared_diff)

        # Calculate the square root to obtain RMSE
        rmse = torch.sqrt(mean_squared_diff)

        return rmse.item()  # Convert to a Python float
    except:
        # Calculate the mean of the squared differences
        mean_squared_diff = np.mean(squared_diff)

        # Calculate the square root to obtain RMSE
        rmse = np.sqrt(mean_squared_diff)

        return rmse

#Training and Testing Loops for Different Cases
def train(get_attn, regression_on, dataloader, model, loss_function, optimizer, device_in_use):
    model.train()

    total_loss=0
    total_correct_1 = 0
    total_samples_1 = 0
    all_targets_1 = []
    all_predictions_1 = []

    attention_ovr_batch = []

    total_rmse = 0

    if not regression_on:
        for (cat_x, cont_x, labels) in dataloader:
            cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

            if get_attn:
                predictions, attention = model(cat_x, cont_x)
                attention_ovr_batch.append(attention.detach().cpu().numpy())
            else:
                predictions = model(cat_x, cont_x)

            loss = loss_function(predictions, labels.long())
            total_loss+=loss.item()

            #computing accuracy
            y_pred_softmax_1 = torch.softmax(predictions, dim=1)
            _, y_pred_labels_1 = torch.max(y_pred_softmax_1, dim=1)
            total_correct_1 += (y_pred_labels_1 == labels).sum().item()
            total_samples_1 += labels.size(0)
            all_targets_1.extend(labels.cpu().numpy())
            all_predictions_1.extend(y_pred_labels_1.cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

        avg_loss = total_loss/len(dataloader)
        accuracy = total_correct_1 / total_samples_1
        f1 = f1_score(all_targets_1, all_predictions_1, average='weighted')

        if get_attn:
            return avg_loss, accuracy, f1, attention_ovr_batch
        else:
            return avg_loss, accuracy, f1
    
    else:
        for (cat_x, cont_x, labels) in dataloader:
            cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

            if get_attn:
                predictions, attention = model(cat_x, cont_x)
                attention_ovr_batch.append(attention.detach().cpu().numpy())
            else:
                predictions = model(cat_x, cont_x)

            loss = loss_function(predictions, labels.unsqueeze(1))
            total_loss+=loss.item()

            rmse_value = rmse(labels.unsqueeze(1), predictions)
            total_rmse+=rmse_value

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

        avg_loss = total_loss/len(dataloader)
        avg_rmse = total_rmse/len(dataloader)

        if get_attn:
            return avg_loss, avg_rmse, attention_ovr_batch
        else:
            return avg_loss, avg_rmse

def test(get_attn, regression_on, dataloader, model, loss_function, device_in_use):
    model.eval()

    total_loss=0
    total_correct_1 = 0
    total_samples_1 = 0
    all_targets_1 = []
    all_predictions_1 = []

    attention_ovr_batch = []

    total_rmse = 0

    if not regression_on:
        with torch.no_grad():
            for (cat_x, cont_x, labels) in dataloader:
                cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

                if get_attn:
                    predictions, attention = model(cat_x, cont_x)
                    attention_ovr_batch.append(attention.detach().cpu().numpy())
                else:
                    predictions = model(cat_x, cont_x)

                loss = loss_function(predictions, labels.long())
                total_loss+=loss.item()

                #computing accuracy
                y_pred_softmax_1 = torch.softmax(predictions, dim=1)
                _, y_pred_labels_1 = torch.max(y_pred_softmax_1, dim=1)
                total_correct_1 += (y_pred_labels_1 == labels).sum().item()
                total_samples_1 += labels.size(0)
                all_targets_1.extend(labels.cpu().numpy())
                all_predictions_1.extend(y_pred_labels_1.cpu().numpy())

                torch.cuda.empty_cache()

            avg_loss = total_loss/len(dataloader)
            accuracy = total_correct_1 / total_samples_1
            f1 = f1_score(all_targets_1, all_predictions_1, average='weighted')

            if get_attn:
                return avg_loss, accuracy, f1, attention_ovr_batch
            else:
                return avg_loss, accuracy, f1

    else:
        with torch.no_grad():
            for (cat_x, cont_x, labels) in dataloader:
                cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

                if get_attn:
                    predictions, attention = model(cat_x, cont_x)
                    attention_ovr_batch.append(attention.detach().cpu().numpy())
                else:
                    predictions = model(cat_x, cont_x)

                loss = loss_function(predictions, labels.unsqueeze(1))
                total_loss+=loss.item()

                rmse_value = rmse(labels.unsqueeze(1), predictions)
                total_rmse+=rmse_value

                torch.cuda.empty_cache()

            avg_loss = total_loss/len(dataloader)
            avg_rmse = total_rmse/len(dataloader)

            if get_attn:
                return avg_loss, avg_rmse, attention_ovr_batch
            else:
                return avg_loss, avg_rmse
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Made with GPT
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, mode='auto'):
        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum change in monitored quantity to qualify as improvement
        self.verbose = verbose  # Whether to print information about the early stopping
        self.mode = mode  # 'auto', 'min', or 'max'

        self.counter = 0  # Counter to keep track of epochs without improvement
        self.best_score = None  # Best validation score
        self.early_stop = False  # Flag to indicate if training should stop

        if self.mode not in ['auto', 'min', 'max']:
            raise ValueError("Mode must be one of 'auto', 'min', or 'max'.")

        if self.mode == 'min':
            self.delta *= -1  # For 'min' mode, reverse the delta direction

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif ((val_score - self.best_score) <= self.delta and self.mode != 'min') or ((val_score - self.best_score) >= self.delta and self.mode == 'min'):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


