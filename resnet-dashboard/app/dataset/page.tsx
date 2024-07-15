import * as React from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';
import Image from 'next/image';

const DatasetDescription = () => {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-100">
      <Card className="w-full max-w-2xl p-6 bg-white rounded-lg shadow-md">
        <CardHeader>
          <CardTitle>About the Dataset</CardTitle>
          <CardDescription>
            Information about the dataset used for training the model.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <h2 className="text-lg font-bold mb-4">Dataset Overview</h2>
          <p className="mb-4">
            The dataset used for training the model is the Fashion MNIST dataset. It consists of 60,000 grayscale images of 10 different fashion categories, with 6,000 images per category.
          </p>
          <ul className="list-disc list-inside mb-4">
            <li>Each image is 28x28 pixels</li>
            <li>10 categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot</li>
          </ul>
          <h2 className="text-lg font-bold mb-4">Category Distribution</h2>
          <p className="mb-4">
            Below is the distribution of sample images in the dataset:
          </p>
          <Image
            src="/dataset.png" // Add the correct path to your image
            alt="Dataset Sample"
            width={600}
            height={400}
            className="mb-4"
          />
          <h2 className="text-lg font-bold mb-4">Data Augmentation</h2>
          <p className="mb-4">
            To improve the model's performance and generalization, data augmentation techniques are applied to the training images. These techniques include:
          </p>
          <ul className="list-disc list-inside mb-4">
            <li>Random rotations</li>
            <li>Random shifts</li>
            <li>Random flips</li>
          </ul>
        </CardContent>
      </Card>
    </main>
  );
};

export default DatasetDescription;
