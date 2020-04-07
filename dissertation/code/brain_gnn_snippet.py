class BrainGNN(torch.nn.Module):
    def __init__(self, conv_type, n_node_features, n_conv_layers, 
                layer_sizes, dropout_p):
        # ...

        # Refer to the layers of the same type through a list.
        self.conv = torch.nn.ModuleList()
        self.fc = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()

        # ...

        # Add convolutional layers.
        size = n_node_features
        for i in range(n_conv_layers):
            if conv_type == ConvTypes.GCN:
                self.conv.append(torch.nn.GCNConv(size, layer_sizes[i]))
            elif conv_type == ConvTypes.GAT:
                self.conv.append(torch.nn.GATConv(size, layer_sizes[i]))
            else:
                self.conv.append(torch.nn.Linear(size, layer_sizes[i]))
            size = layer_sizes[i]
        
        # Add remaining fully connected layers.
        for i in range(len(layer_sizes) - n_conv_layers):
            self.fc.append(torch.nn.Linear(size, layer_sizes[n_conv_layers+i]))
            size = layer_sizes[n_conv_layers+i]
            if i < len(layer_sizes) - n_conv_layers - 1:
                self.dropout.append(torch.nn.Dropout(dropout_p))

    # Forward propagation shows how the layers are applied.
    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        for i in range(len(self.conv)):
            x = self.conv[i](x, edge_index)
            x = torch.tanh(x)
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = torch.tanh(x)
            x = self.dropout[i](x)

        # Apply the last fully connected layer without the activation function or dropout.
        x = self.fc[-1](x)
        return x
