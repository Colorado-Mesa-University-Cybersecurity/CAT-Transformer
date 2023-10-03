import torch
import torch.nn as nn
from rff.layers import GaussianEncoding
from sklearn.metrics import accuracy_score, f1_score, recall_score

# All the layers of the model

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

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) #(batch_size, head_dim, #query_embeddings, #key_embeddings)

        # Calculate simplified attention scores
        avg_attention = attention.mean(dim=0)  # Average across batches
        # print("batch average", avg_attention.shape)
        avg_attention = avg_attention.mean(dim=0).squeeze(dim=0)
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
                                          nn.ReLU(),
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
                 pre_norm_on
    ):
        super(Decoder, self).__init__()

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
        self.avg_attention = None

    def forward(self, class_embed, context):
        for layer in self.layers:
            # x is the classification embedding (CLS Token)
            # context are the feature embeddings that will be used as key and value
            x, self.avg_attention = layer(class_embed, context, context)
  
        return x 

class Embeddings(nn.Module):
    def __init__(self, sigma, embed_size, input_size, embedding_dropout, n_features, num_target_labels, rff_on):
        super(Embeddings, self).__init__()

        self.rff_on = rff_on

        if self.rff_on:
            self.rffs = nn.ModuleList([GaussianEncoding(sigma=sigma, input_size=input_size, encoded_size=embed_size//2) for _ in range(n_features)])
            self.dropout = nn.Dropout(embedding_dropout)
            self.mlp_in = embed_size
        else:
            self.mlp_in = input_size

        self.embeddings = nn.ModuleList([nn.Linear(in_features=self.mlp_in, out_features=embed_size) for _ in range(n_features)])

        # Classifcation Embeddings for each target label
        self.target_label_embeddings = nn.ModuleList([nn.Embedding(1, embed_size) for _ in range(num_target_labels)])


    def forward(self, x):
        x = x.unsqueeze(2) #(batch_size, n_features) -> (batch_size, n_features, 1)
        rff_vectors = []
        if self.rff_on:
            for i, r in enumerate(self.rffs):
                input = x[:,i,:]
                out = r(input)
                rff_vectors.append(out)
        
            x = torch.stack(rff_vectors, dim=1)
        
        embeddings = []
        for i, e in enumerate(self.embeddings):
            goin_in = x[:,i,:]
            goin_out = e(goin_in)
            embeddings.append(goin_out)

        target_label_embeddings_ = []
        for e in self.target_label_embeddings:
            input = torch.tensor([0], device=x.device)
            temp = e(input)
            temp = temp.repeat(x.size(0), 1)
            tmep = temp.unsqueeze(1)
            target_label_embeddings_.append(temp)

        class_embeddings = torch.stack(target_label_embeddings_, dim=1)
        
        # class_embed = self.classification_embedding(torch.tensor([0], device=x.device))  # use index 0 for the classification embedding
        # class_embed = class_embed.repeat(x.size(0), 1) # -> (batch_size, embed_size)
        # class_embed = class_embed.unsqueeze(1)

        context = torch.stack(embeddings, dim=1)

        return class_embeddings, context

class classificationHead(nn.Module):
    def __init__(self, embed_size, dropout, mlp_scale_classification, num_target_classes):
        super(classificationHead, self).__init__()
        
        #flattening the embeddings out so each sample in batch is represented with a 460 dimensional vector
        self.input = embed_size
        self.lin1 = nn.Linear(self.input, mlp_scale_classification*self.input)
        self.drop = nn.Dropout(dropout)
        # self.lin2 = nn.Linear(2*self.input, 2*self.input)
        self.lin3 = nn.Linear(mlp_scale_classification*self.input, self.input)
        self.lin4 = nn.Linear(self.input, num_target_classes)
        self.relu = nn.ReLU()
        self.initialize_weights()

    def initialize_weights(self): #he_initialization.
        torch.nn.init.kaiming_normal_(self.lin1.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.lin1.bias)

        torch.nn.init.kaiming_normal_(self.lin3.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.lin3.bias)

    def forward(self, x):

        x= torch.reshape(x, (-1, self.input))

        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin4(x)
  
        return x


# DEFAULT PARAMETERS SET UP FOR MY DATASET. BE CAREFUL AND MAKE SURE YOU SET THEM UP HOW YOU WANT.
# All dropout is initially turned off
class Classifier(nn.Module):
    def __init__(self, 
                 rff_on = False,
                 sigma=4,
                 embed_size=20,
                 input_size=1,
                 embedding_dropout = 0,
                 n_features=66, # YOU WILL PROBABLY NEED TO CHANGE
                 num_layers=1,
                 heads=1,
                 forward_expansion=4, # Determines how wide the MLP is in the encoder. Its a scaling factor. 
                 decoder_dropout=0,
                 classification_dropout = 0,
                 pre_norm_on = False,
                 mlp_scale_classification = 4, #widens the mlp in the classification heads
                 targets_classes : list= [0] #[12,2] #put the number of classes in each target variable. traffic type = 3 classes, application type = 8 classes
                 ):
        super(Classifier, self).__init__()

        self.embeddings = Embeddings(rff_on=rff_on, sigma=sigma, embed_size=embed_size, input_size=input_size, embedding_dropout=embedding_dropout, n_features=n_features, num_target_labels=len(targets_classes))
        self.decoder = Decoder(embed_size=embed_size, num_layers=num_layers, heads=heads, forward_expansion=forward_expansion, decoder_dropout=decoder_dropout, pre_norm_on=pre_norm_on)
        self.classifying_heads = nn.ModuleList([classificationHead(embed_size=embed_size, dropout=classification_dropout, mlp_scale_classification=mlp_scale_classification, num_target_classes=x) for x in targets_classes])
        
    def forward(self, x):
        class_embed, context = self.embeddings(x)

        x = self.decoder(class_embed, context)
        
        probability_dist_raw = []
        for i, e in enumerate(self.classifying_heads):
            input = x[:, i,:]
            output = e(input)
            probability_dist_raw.append(output)
        
        return probability_dist_raw


# Training and Testing Loops
# Should not need modification
def train(dataloader, model, loss_functions : list, optimizer, device_in_use):
  total_loss = 0
  target_count = len(loss_functions)

  total_correct = [0] * target_count
  total_samples = [0] * target_count
  all_targets   = [[] for _ in range(target_count)]
  all_predictions = [[] for _ in range(target_count)]
  y_pred_softmax = [0] * target_count
  y_pred_labels = [0] * target_count
  accuracy = []

  model.train()

  for (inputs, targets) in dataloader:
      
    inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
    task_predictions = model(inputs) #contains a list of the tensor outputs for each task

    loss = 0
    for i in range(len(loss_functions)):
      loss += loss_functions[i](task_predictions[i], targets[:,i])
    loss = loss/len(loss_functions)
    total_loss+= loss.item() #just summing the two losses
    
    for i in range(target_count):
        y_pred_softmax[i] = torch.softmax(task_predictions[i], dim=1)
        _, y_pred_labels[i] = torch.max(y_pred_softmax[i], dim=1)
        total_correct[i] += (y_pred_labels[i] == targets[:,i]).sum().item()
        total_samples[i] += targets[:,i].size(0)
        all_targets[i].extend(targets[:,i].cpu().numpy())#ask warrin about this section
        all_predictions[i].extend(y_pred_labels[i].cpu().numpy())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  #Loss
  avg_loss = total_loss/len(dataloader)
  #Accuracies
  for x in range(target_count):
     accuracy.append(total_correct[x]/total_samples[x])


  return avg_loss, accuracy





def test(dataloader, model, loss_functions : list, device_in_use):
  total_loss = 0
  target_count = len(loss_functions)
  total_correct = [0] * target_count
  total_samples = [0] * target_count
  all_targets   = [[] for _ in range(target_count)]
  all_predictions = [[] for _ in range(target_count)]
  y_pred_softmax = [0] * target_count
  y_pred_labels = [0] * target_count
  accuracy = []
  f1       = []

  model.eval()

  with torch.no_grad():
    for (inputs, targets) in dataloader:
      inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
      
      #compute prediction error
      task_predictions = model(inputs) #contains a list of the tensor outputs for each task

      loss = 0
      for i in range(len(loss_functions)):
        loss += loss_functions[i](task_predictions[i], targets[:,i])
      loss = loss/len(loss_functions)
      total_loss+= loss.item()

      for i in range(target_count):
        y_pred_softmax[i] = torch.softmax(task_predictions[i], dim=1)
        _, y_pred_labels[i] = torch.max(y_pred_softmax[i], dim=1)
        total_correct[i] += (y_pred_labels[i] == targets[:,i]).sum().item()
        total_samples[i] += targets[:,i].size(0)
        all_targets[i].extend(targets[:,i].cpu().numpy())#ask warrin about this section
        all_predictions[i].extend(y_pred_labels[i].cpu().numpy())

  #Loss
  avgloss = total_loss/len(dataloader)
  #Accuracies
  for x in range(target_count):
     accuracy.append(total_correct[x]/total_samples[x])
  # for x,y in total_correct, total_samples:
  #   accuracy.append(x/y)
  #F1 Score  
  for z,w in zip(all_targets, all_predictions):
    f1.append(f1_score(z , w, average='weighted'))
  #Recall
  # recall = recall_score(all_targets, all_predictions, average='weighted')


  return avgloss, accuracy, all_predictions, all_targets, f1
