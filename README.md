# Security Attacks on Machine Learning Systems

## Overview
This project demonstrates the vulnerability of deep learning models to adversarial attacks using the MNIST dataset. By making subtle, imperceptible changes to input images, we can cause a neural network to misclassify handwritten digits while remaining visually identical to humans. Our implementation achieved a 97.10% success rate while modifying only 4.02% of input features.

## Features
- Interactive interface with multiple testing options
- Support for both MNIST dataset and custom hand-drawn inputs
- Real-time visualization of adversarial attacks
- Comprehensive success rate analysis and confidence metrics
- Drawing interface support for custom digit input

## Requirements
- Python 3.8+ (Tested on 3.9-3.11.2)
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Installation
```bash
# Clone the repository
git clone https://github.com/mchapaton/adversarial-attack.git

# Navigate to the project directory
cd adversarial-attack

# Install required dependencies
pip3 install tensorflow numpy matplotlib pillow

# Run the program
python main.py
```

## Project Structure
```
project/
├── adversarial.py       # Core attack implementations
├── data_preprocessing.py # Data handling
├── evaluate_model.py    # Model evaluation functions
├── main.py             # Interactive demo interface
├── model.py            # Neural network architectures
└── train_model.py      # Model training implementation
```

## Usage
Run the interactive demo:
```bash
python main.py
```

The system offers four main options:
1. Test MNIST digits with Saliency map (iterate epsilon and top N)
2. Test uploaded image with Saliency map (iterate epsilon and top N)
3. Test MNIST digits with Saliency map (specify epsilon and top N)
4. Test uploaded image with Saliency map (specify epsilon and top N)

## Implementation Details
- Uses gradient-based and saliency map approaches for generating adversarial examples
- Supports both FGSM (Fast Gradient Sign Method) and targeted pixel modifications
- Interactive visualization of attack success rates and perturbation effects
- Real-time demonstration capabilities using Raspberry Pi and Wacom tablet

## Results
- 97.10% success rate in generating adversarial examples
- Only 4.02% of input features modified on average
- Human study confirmed adversarial examples remain visually indistinguishable
- Identified specific digit pairs more susceptible to attacks (e.g., '1' -> '7')

## Hardware Setup
For the demonstration setup:
- Raspberry Pi 5
- Wacom Drawing Tablet
- 2x Portable Monitors

## Contributing
This project was developed as part of the ECE Senior Design course at Michigan State University. While we're not actively maintaining it past FS24, feel free to fork and extend the project.

## Team Members
- Rashed Almualla - Document Prep, Software
- Mathieu Chapaton - Management, Software
- Blake Morris - Lab Coordinator, Software/Hardware
- Faris Sweis - Presentation Prep, Software/Hardware

## Acknowledgments
- Prof. Jian Ren - Project Facilitator
- Prof. Subir Biswas - Course Professor
- Based on research paper: "The Limitations of Deep Learning in Adversarial Settings" (2016)
- MNIST Dataset
