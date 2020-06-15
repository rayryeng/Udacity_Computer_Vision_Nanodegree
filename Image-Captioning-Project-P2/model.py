import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # Create internal variables + dropout probability
        self.dropout_prob = 0.3
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_size = embed_size

        # Define LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Define embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        # Define dropout layer between output of LSTM to linear
        self.dropout = nn.Dropout(self.dropout_prob)

        # Define linear layer for outputs
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):        
        # Convert one-hot word tensors into embedding matrix
        # Do not include the last column as that is the end token
        # and we are predicting the sequence from 0 up to the second
        # last token
        # Embedding --> batch_size x caption_size - 1 becomes batch_size x caption_size - 1 x embed_size
        embed = self.embed(captions[:, :-1])

        # Must combine the input features and word embeddings together
        # The first input are the input features
        # After that follow the word embeddings
        # embed[0] is the first image
        # embed[0, 0] is the image feature vector from the encoder for the first image
        # embed[0, j] if j >= 1 is the embedding vector for the jth word of the first image
        # embed[i, j] is the jth embedding vector for the jth caption word of the ith image
        # features becomes batch_size x 1 x embed_size
        # captions --> batch_size x caption_size - 1 x embed_size
        # Final size --> batch_size x caption_size x embed_size
        embed = torch.cat([features.unsqueeze(1), embed], 1)

        # Supply these features into the LSTM
        # Hidden states should be initialized to zero so not supplying
        # the hidden states default to this
        # Output size is batch_size x caption_size x hidden_size
        r_output, _ = self.lstm(embed)
        
        # Pass through a dropout layer
        out = self.dropout(r_output)
        
        # Put through the fully-connected layer
        # If we have a 3d matrix, the linear layer is applied to the
        # last dimension of the input tensor
        # Input - batch_size x caption_size x hidden_size
        # Output - batch_size x caption_size x vocab_size
        out = self.fc(out)
        
        # Now return
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        for _ in range(max_len):
            # Perform RNN processing chain
            # Remember that the input initially is all zeroes
            # Pipe through LSTM
            # Inputs --> 1 x 1 x embed_size
            # Second dim has been unsqueezed from 3_Inference notebook
            out, states = self.lstm(inputs, states)

            # Pipe through linear layer
            # Input size - 1 x 1 x hidden_size
            # Output size - 1 x 1 x vocab_size
            out = self.fc(out)

            # Figure out the ID / location of the word to choose
            index = out.argmax(2)
            output.append(out.argmax(2).item())

            # Feed the output into the embedding layer as input and try again
            # index is a scalar but because of argmax as we have fed in a 1 x 1 x N
            # tensor, we thus get a 1 x 1 scalar (2D with singleton dimensions)
            # So the index is 1 x 1
            # batch_size = 1, caption_size = 1
            # Output thus becomes 1 x 1 x embed_size to feed back into the LSTM
            inputs = self.embed(index)

        return output