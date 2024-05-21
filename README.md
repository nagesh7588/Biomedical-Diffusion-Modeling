# LEI Diffusion: Generating Least Excitable Images

## Overview
This project demonstrates a modification to the diffusion model for generating Least Excitable Images (LEIs) using the Energy Guided Generative (EGG) process. LEIs are images designed to evoke minimal neural activation compared to traditional Most Excitable Images (MEIs), which aim to maximize neural response.

## Installation
1. Clone this repository to your local machine:
   ```
   git clone https://github.com/your_username/LEI-diffusion.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Modify the `energy_function` in `diffusion.py` to reflect the energy function of your target neural network model.
2. Use the `generate_mei` function in `diffusion.py` to generate MEIs.
3. Use the `generate_lei` function in `diffusion.py` to generate LEIs.

## Example
```python
import torch
from diffusion import generate_mei, generate_lei

# Example usage
initial_image = torch.randn(1, 3, 224, 224)  # Random initial image
target_neuron = 0  # Example neuron index

mei = generate_mei(initial_image, target_neuron)
lei = generate_lei(initial_image, target_neuron)
```

## Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust and expand this README to fit the specific details and requirements of your project!
