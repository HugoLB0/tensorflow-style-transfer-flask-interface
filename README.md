# TensorFlow Style Transfer

This project combines the power of TensorFlow for neural style transfer and provides a Flask web interface for applying artistic styles to images. Below, we describe the functionalities of this project.


https://github.com/HugoLB0/tensorflow-style-transfer-flask-interface/assets/66400773/7b793519-6e86-4cc5-b6d0-59b954a12365


## Table of Contents
- [Introduction](#introduction)
- [Functionalities](#functionalities)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project allows users to apply artistic styles to their images using neural style transfer. It utilizes a pre-trained VGG model to extract content and style features from images and combines them to generate stylized images. The Flask web interface makes it easy for users to upload content and style images and receive the stylized output.

## Functionalities

1. **Style Transfer**: Users can upload a content image and a style image through the web interface. The project uses neural style transfer to create a stylized image that combines the content of the content image with the artistic style of the style image.

2. **Customization**: Users can adjust the content and style weights to control the balance between preserving the content of the content image and applying the style of the style image.

3. **Logging**: The project includes logging capabilities to track the progress of style transfer, including the chosen weights, model details, and processing time.

4. **Output**: The final stylized image is displayed on the web interface for users to download.

Please refer to the code and comments in the provided code snippet for a more detailed understanding of how these functionalities are implemented.

## Getting Started

To get started with this TensorFlow Style Transfer project, follow these steps:

1. Ensure you have TensorFlow and other dependencies installed. You can install them using `pip`.

2. Clone the repository to your local machine.

3. Run the Flask application by executing the Python script. Typically, this is done with the following command:

   ```bash
   python app.py
4. Access the Flask web interface in your web browser at http://localhost:5000.

## Usage
Upload a content image and a style image.

Adjust the content and style weights if desired.

Initiate the style transfer process.

Download the stylized output image once the process is complete.

## Customization
You can customize this project by modifying the neural style transfer parameters, such as the content and style layers, weights, and optimization settings, in the code.

## Contributing
Contributions to this project are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Thank you for using our TensorFlow Style Transfer project! If you have any questions or need assistance, please don't hesitate to reach out.

Happy style transferring! 🎨
