import * as React from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';
import Image from 'next/image';

const ModelDescription = () => {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-100">
      <Card className="w-full max-w-2xl p-6 bg-white rounded-lg shadow-md">
        <CardHeader>
          <CardTitle>About the Model</CardTitle>
          <CardDescription>
            Detailed information about the neural network model used for predictions.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <h2 className="text-lg font-bold mb-4">Model Architecture</h2>
          <p className="mb-4">
            The model is a Convolutional Neural Network (CNN) enhanced with residual blocks, inspired by ResNet architecture <a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">[He et al., 2015]</a>. The architecture helps mitigate the vanishing gradient problem and allows for deeper networks by including shortcut connections that bypass one or more layers.
          </p>
          <ul className="list-disc list-inside mb-4">
            <li>Input layer with shape (28, 28, 1)</li>
            <li>Convolutional layer with 32 filters and ReLU activation</li>
            <li>MaxPooling layer</li>
            <li>Residual blocks with convolutional and batch normalization layers</li>
            <li>Dropout layers to prevent overfitting</li>
            <li>Fully connected layers with ReLU activation</li>
            <li>Output layer with 10 classes and softmax activation</li>
          </ul>
          <Image
            src="/model_res.png" // Add the correct path to your image
            alt="Model Architecture"
            width={600}
            height={400}
            className="mb-4"
          />
          <h2 className="text-lg font-bold mb-4">Residual Connections</h2>
          <p className="mb-4">
            The residual blocks in the model consist of two convolutional layers with batch normalization and ReLU activation, followed by a shortcut connection that bypasses these layers. This structure allows the gradient to flow directly through the shortcut connections during backpropagation, improving training efficiency and performance.
          </p>
          <Image
            src="/residual.png" // Add the correct path to your image
            alt="Residual Block"
            width={600}
            height={400}
            className="mb-4"
          />
          <h2 className="text-lg font-bold mb-4">Optimizer</h2>
          <p className="mb-4">
            The model uses the Adam optimizer with gradient clipping to prevent exploding gradients. The loss function is sparse categorical cross-entropy, and the performance metric is accuracy.
          </p>
          <h2 className="text-lg font-bold mb-4">Training and Validation</h2>
          <p className="mb-4">
            The model was trained on the Fashion MNIST dataset, which consists of 70,000 images. The training set has 60,000 images, and the validation set has 10,000 images. Below is the training history showing the model's accuracy and loss over epochs.
          </p>
          <Image
            src="/training.png" // Add the correct path to your image
            alt="Training History"
            width={600}
            height={400}
            className="mb-4"
          />
          <h2 className="text-lg font-bold mb-4">Confusion Matrix</h2>
          <p className="mb-4">
            The confusion matrix below shows the performance of the model on the test dataset, providing insights into the model's accuracy for each class.
          </p>
          <Image
            src="/confusion_matrix.png" // Add the correct path to your image
            alt="Confusion Matrix"
            width={600}
            height={400}
            className="mb-4"
          />
        </CardContent>
      </Card>
    </main>
  );
};

export default ModelDescription;
