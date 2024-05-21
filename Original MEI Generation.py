import torch
import torch.nn.functional as F

# Assuming model is a pre-trained neural network
model = ...  # Load your model here
model.eval()

def energy_function(x, target_neuron):
    # Get the activation of the target neuron
    activation = model(x)[0, target_neuron]
    return activation

def generate_mei(initial_image, target_neuron, num_steps=100, lr=0.01):
    x = initial_image.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        energy = energy_function(x, target_neuron)
        loss = -energy  # We maximize the activation
        loss.backward()
        optimizer.step()
    
    return x.detach()

# Example usage
initial_image = torch.randn(1, 3, 224, 224)  # Random initial image
target_neuron = 0  # Example neuron index
mei = generate_mei(initial_image, target_neuron)
