class BrainGNN(torch.nn.Module):
    def __init__(self, conv_type, n_node_features, n_conv_layers, 
                layer_sizes, dropout_p):

        # Refer to the layers of the same type through a list.
        self.conv = torch.nn.ModuleList() # Convolutional layers.
        self.fc = torch.nn.ModuleList() # Fully connected layers.
        self.dropout = torch.nn.ModuleList() # Dropout layers.

        # Add convolutional layers of conv_type
        # ...
        # Add remaining fully connected and dropout layers.
        # ...       

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
