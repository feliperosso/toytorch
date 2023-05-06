""" 
Full Models 

This Module contains several fully working Neural Network models, using only
the layers and functions previously built in the modules nnLayers and nnFunc:
    - A Convolutional Neural Network which trains to classify
    the CIFAR10 image dataset.
    - An LSTM Language Model that can train on any text and then be used
    to generate new text in the style of the training dataset.
    - A Transformer Decoder only model that can train on any text and then
    be used to generate new text in the style of the training dataset.

The only torch.nn commands used in this file are:
    - torch.nn.Module
    - torch.nn.ModuleList
"""

# Load Packages
import torch, math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
# Load Modules
from toytorch.func import softmax, cosine_schedule, cross_entropy, one_hot, relu
from toytorch.layers import (Embedding, LSTM, Dropout, PositionalEncoding,
                             TransformerEncoderLayer, Convolution, Pooling, Linear)

# - Convolutional CIFAR10 Model (Classification) -
class Conv_CIFAR10(torch.nn.Module):

    def __init__(self, n_channels, num_classes, dropout_rate):
        super().__init__()
        # Define and initialize layers
        self.conv1 = Convolution(3, n_channels[0], (3, 3), (32, 32), stride=1, padding=1)
        self.conv2 = Convolution(n_channels[0], n_channels[0], (3, 3), (32, 32), stride=1, padding=1)
        self.pool1 = Pooling((2, 2), (32, 32), stride=2)

        self.conv3 = Convolution(n_channels[0], n_channels[1], (3, 3), (16, 16), stride=1, padding=1)
        self.conv4 = Convolution(n_channels[1], n_channels[1], (3, 3), (16, 16), stride=1, padding=1)
        self.pool2 = Pooling((2, 2), (16, 16), stride=2)

        self.linear1 = Linear(n_channels[1]*8*8, 1024)
        self.linear2 = Linear(1024, num_classes)

        self.dropout = Dropout(dropout_rate)
        # Number of learning parameters
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of parameters in model: {self.num_parameters:,}')
        
    def forward(self, input):
        """ input: (batch_size, in_channels, in_height, in_width)
            output: (batch_size, num_classes) """
        # Convolutional layers
        out = relu(self.conv1(input))
        out = relu(self.conv2(out))
        out = self.dropout(self.pool1(out))

        out = relu(self.conv3(out))
        out = relu(self.conv4(out))
        out = self.dropout(self.pool2(out)).view(input.shape[0], -1)
        # Feedforward layers
        out = self.dropout(relu(self.linear1(out)))
        out = self.linear2(out)
        return out
    
    @torch.no_grad()
    def accuracy(self, data, batch_size, device):
        """ data: is either test, val or train data """
        self.to(device)
        self.eval()
        X, Y = data
        # Split into batches in order to avoid overflowing CUDA memory
        length = len(X)
        # Reshaped dimensions
        X_reshape = tuple([length//batch_size, batch_size] + list(X.shape[-3:]))
        Y_reshape = tuple([length//batch_size, batch_size])
        
        x_batches = X.reshape(X_reshape)
        y_batches = Y.reshape(Y_reshape)

        accuracy = 0
        for x, y in zip(x_batches, y_batches):
            x, y = x.to(device), y.to(device) 
            # Compute accuracy
            logits = self(x)
            nn_pred = torch.argmax(logits, dim=-1)
            accuracy += torch.sum(nn_pred == y).item()
            torch.cuda.empty_cache()
        accuracy = accuracy*100/length
        return accuracy
    
    def train_model(self, data, optimizer, epochs:int, batch_size, device,
                    learning_schedule=False, save=None):
        """ If learning_schedule=True a Cosine schedule is applied, so that eta_fin = eta_in/10
            If save = string, then the model with the best validation accuracy is saved """
        self.to(device)
        # Split data types
        train_data, validation_data, test_data = data
        # Further split train data
        X_train, Y_train = train_data
        n_train = len(X_train)
        # Training Loss
        global_training_time, stochastic_training_loss = [], []
        t_global = 0
        # Validation Accuracy
        validation_accuracy = []
        best_epoch, best_validation_accuracy = 0, 0
        # Learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        # Reshaped dimensions depending on n_batch
        num_batches = n_train//batch_size
        X_reshape_batch_split = tuple([num_batches, batch_size] + list(X_train.shape[-3:]))
        Y_reshape_batch_split = tuple([num_batches, batch_size])

        prog_bar = tqdm(range(epochs), total=epochs)
        for t_epoch in prog_bar:
            self.train()
            # Shuffle and divide into batches
            shuffle = torch.randperm(n_train)
            X_train, Y_train = X_train[shuffle], Y_train[shuffle]
            x_batches = X_train.reshape(X_reshape_batch_split)
            y_batches = Y_train.reshape(Y_reshape_batch_split) 
            # Loop over batches
            for x, y in zip(x_batches, y_batches):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                t_global += 1
                # Forward
                logits = self(x)
                y_prob = one_hot(y, 10)
                # Compute loss
                loss = cross_entropy(logits, y_prob)
                # Backprop
                loss.backward()
                optimizer.step()
                # Store training loss
                global_training_time.append(t_global)
                stochastic_training_loss.append(loss.item())
                # Progress bar
                prog_bar.set_description(f'Train Loss: {loss.item():.2f}. Best Validation Accuracy/Epoch: {best_validation_accuracy:.2f}/{best_epoch}')
                # Learning Schedule
                if learning_schedule:
                    optimizer.param_groups[0]['lr'] = cosine_schedule(initial_lr, t_global, t_max=num_batches*epochs)
                torch.cuda.empty_cache()
            
            # Compute validation accuracy
            validation_accuracy.append(self.accuracy(validation_data, batch_size, device))
            if best_validation_accuracy < validation_accuracy[-1]:
                best_validation_accuracy = validation_accuracy[-1]
                best_epoch = t_epoch
                if save: 
                    torch.save(self.state_dict(), save + '.pt')
        
        # Load model with best validation accuracy
        if save:
            self.load_state_dict(torch.load(save + '.pt'))
        # Compute final test accuracy
        final_test_accuracy = self.accuracy(test_data, batch_size, device)
        # Plot training loss
        plt.semilogy(global_training_time, stochastic_training_loss)
        plt.xlabel("Iterations")
        plt.ylabel("Training Loss")
        plt.title(f'Final Test Accuracy: {final_test_accuracy:.2f}')
        plt.show()   
        
        return global_training_time, stochastic_training_loss, validation_accuracy

# - LSTM Language Model -
class LSTMLanguageModel(torch.nn.Module):

    def __init__(self, vocab_dim, embed_dim, num_layers, dropout_rate):
        super().__init__()
        # Parameters
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        # Initialize Layers. (hidden_dim == embed_dim)
        self.embedding = Embedding(vocab_dim, embed_dim)
        self.LSTMs = torch.nn.ModuleList(LSTM(embed_dim, embed_dim) for _ in range(num_layers))
        self.dropout = Dropout(dropout_rate)
        # Number of learning parameters
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of parameters in model: {self.num_parameters:,}')
    
    def forward(self, input, h0l, c0l):
        """
        ARGS:
            - input: (sequence_length, batch_size) torch tensor.
            - h0l, c0l: (num_layers, batch_size, hidden_dim).
        OUTPUT:
            - output: (sequence_length, batch_size, vocabulary_size)
            - h_final, c_final: (num_layers, batch_size, hidden_dim)
        """
        h_final = torch.zeros(self.num_layers, input.shape[-1], self.embed_dim, device=input.device)
        c_final = torch.zeros(self.num_layers, input.shape[-1], self.embed_dim, device=input.device)
        # Forward
        output = self.embedding(input)
        # LSTMs
        for num_layer, LSTM, h0, c0 in zip(torch.arange(self.num_layers), self.LSTMs, h0l, c0l):
            output, (h_f, c_f) = LSTM(output, h0, c0)
            h_final[num_layer] = h_f
            c_final[num_layer] = c_f
        output = self.dropout(output)
        output = output@self.embedding.embedding_matrix.T
        return output, (h_final, c_final)
    
    def train_model(self, train_loader, optimizer, epochs:int, device, learning_schedule=False):
        self.train()
        self.to(device)

        num_batches = len(train_loader)
        training_tokens, training_loss = [], []
        total_tokens = 0

        initial_lr = optimizer.param_groups[0]['lr']
        t_global = 0

        for t_epoch in range(epochs):
            prog_bar = tqdm(enumerate(train_loader), total=num_batches)
            for t_batch, (x_b, y_b) in prog_bar:
                t_global += 1
                # Move to device
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                # Forward (num_LSTM_layers, batch_size, hidden_dim)
                h0l, c0l = torch.zeros(2, self.num_layers, y_b.shape[0], self.embed_dim, device=device)
                logits, _ = self(x_b.T, h0l, c0l)
                y_prob = one_hot(y_b.T, logits.shape[-1])
                # Reshape
                logits = logits.view(-1, logits.shape[-1])
                y_prob = y_prob.view(-1, logits.shape[-1])
                # Compute loss
                loss = cross_entropy(logits, y_prob)
                # Store training loss
                total_tokens += logits.shape[0]*logits.shape[1]
                training_tokens.append(total_tokens)
                training_loss.append(loss.item())
                # Backprop
                loss.backward()
                optimizer.step()
                # Progress bar
                prog_bar.set_description(f'Epoch {t_epoch + 1}. Train Loss: {loss.item():.2f}.')
                # Learning Schedule update (if required)
                if learning_schedule:
                    optimizer.param_groups[0]['lr'] = cosine_schedule(initial_lr, t_global, t_max=num_batches*epochs)

        # Plot training loss
        plt.semilogy(training_tokens, training_loss, label='Training Loss')
        plt.xlabel('Training Tokens')
        plt.title(f'Final Training Perplexity: {round(math.exp(training_loss[-1]), 2)}')
        plt.legend()
        plt.show()

        return training_tokens, training_loss
    
    @torch.no_grad()
    def generate_text(self, enc_prompt, num_tokens:int, temp:int, device):
        self.eval()
        self.to(device)

        # Encode prompt and generate first prediction
        x = enc_prompt.to(device)
        h0l, c0l = torch.zeros(2, self.num_layers, 1, self.embed_dim, device=device)
        output, (h_final, c_final) = self(x[:, None], h0l, c0l)

        pred_prob = softmax(output[-1, 0], dim=-1, temp=temp)
        x_new = torch.multinomial(pred_prob, 1)
        x = torch.cat((x,x_new))

        # Use hidden states to generate tokens
        for _ in range(num_tokens - 1):
            output, (h_final, c_final) = self(x[-1, None], h_final, c_final)
            pred_prob = softmax(output[0], dim=-1, temp=temp)
            x_new = torch.multinomial(pred_prob, 1)
            x = torch.cat((x,x_new))

        return x.tolist() 

# - Transformer Decoder Only (Language Model) -
class TransformerDecoderOnly(torch.nn.Module):

    def __init__(self, vocab_dim, embed_dim, num_layers, num_heads, dim_ff, dropout_rate):
        super().__init__()
        # Initialize layers
        self.embedding = Embedding(vocab_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_encoder = torch.nn.ModuleList(
                                   TransformerEncoderLayer(embed_dim, num_heads, dim_ff, dropout_rate)
                                   for _ in range(num_layers))
        self.dropout = Dropout(dropout_rate)
        # Number of learning parameters
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of parameters in model: {self.num_parameters:,}')

    def forward(self, input, mask=None):
        """ input: (batch_size, sequence_length)
            out: (batch_size, sequence_length, vocab_dim)"""
        # Masking
        if mask == True:
            seq_len = input.shape[-1]
            mask = torch.triu(-torch.ones(seq_len, seq_len)*float('inf'), diagonal=1)
            mask = mask.to(input.device)
        # Forward
        out = self.dropout(self.positional_encoding(self.embedding(input)))
        for single_layer in self.transformer_encoder:
            out, _ = single_layer(out, mask)
        out = out@self.embedding.embedding_matrix.T
        return out
    
    def train_model(self, train_loader, optimizer, epochs:int, device, learning_schedule=False, save=None):
        """ If save is not None, it should be a list with two elements.
            save[0]:str with name of the saved file
            save[1]:int number of iterations the model is saved"""
        self.train()
        self.to(device)

        num_batches = len(train_loader)
        training_tokens, training_loss = [], []
        total_tokens = 0

        initial_lr = optimizer.param_groups[0]['lr']
        t_global = 0

        for t_epoch in range(1, epochs + 1):
            prog_bar = tqdm(enumerate(train_loader), total=num_batches)
            for t_batch, (x_b, y_b) in prog_bar:
                # Move to device
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                # Forward
                logits = self(x_b, mask=True)
                y_prob = one_hot(y_b, logits.shape[-1])
                # Reshape
                logits = logits.view(-1, logits.shape[-1])
                y_prob = y_prob.view(-1, y_prob.shape[-1])
                # Compute Loss
                loss = cross_entropy(logits, y_prob)
                # Backprop
                loss.backward()
                optimizer.step()
                # Store training loss
                total_tokens += x_b.shape[0]*x_b.shape[1]
                training_tokens.append(total_tokens)
                training_loss.append(loss.item())
                # Save model
                if save is not None:
                    if t_batch % save[1] == 0:
                        torch.save(self.state_dict(), save[0] + '.pt')
                # Progress bar
                prog_bar.set_description(f'Epoch {t_epoch}. Train Loss: {loss.item():.2f}.')
                # Learning Schedule update (if required)
                if learning_schedule:
                    optimizer.param_groups[0]['lr'] = cosine_schedule(initial_lr, t_global, t_max=num_batches*epochs)

        # Plot training loss
        plt.semilogy(training_tokens, training_loss, label='Training Loss')
        plt.xlabel('Training Tokens')
        plt.title(f'Final Training Perplexity: {round(math.exp(training_loss[-1]), 2)}')
        plt.legend()
        plt.show()

        return training_tokens, training_loss
    
    @torch.no_grad()
    def generate_text(self, enc_prompt, num_tokens:int, sequence_length:int, temp:int, device):
        self.eval()
        self.to(device)
        x = enc_prompt.to(device)
        for _ in range(num_tokens):
            if x.shape[-1] <= sequence_length:
                context = x
            else:
                context = x[-sequence_length:]
            output = self(context[None, :])
            pred_prob = softmax(output[0, -1], dim=-1, temp=temp)
            x_new = torch.multinomial(pred_prob, 1)
            x = torch.cat((x,x_new))
        return x.tolist()