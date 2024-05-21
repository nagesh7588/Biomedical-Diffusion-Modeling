def generate_lei(initial_image, target_neuron, num_steps=100, lr=0.01):
    x = initial_image.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        energy = energy_function(x, target_neuron)
        loss = energy  # We minimize the activation
        loss.backward()
        optimizer.step()
    
    return x.detach()

# Example usage
initial_image = torch.randn(1, 3, 224, 224)  # Random initial image
target_neuron = 0  # Example neuron index
lei = generate_lei(initial_image, target_neuron)
