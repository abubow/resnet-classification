from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
import numpy as np
import tensorflow as tf
import os
class PredictView(APIView):
    def post(self, request):
        file = request.FILES['image']
        file_name = default_storage.save(file.name, ContentFile(file.read()))
        file_path = default_storage.path(file_name)
        
        # Load and preprocess the image
        image = Image.open(file_path).convert('L')
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)

        # Load the model
        if os.path.exists('../mnist_resnet/model.keras'):
            model = tf.keras.models.load_model('../mnist_resnet/model.keras')
        else:
            print('Model not found. Please train the model first.')
        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Clean up the saved file
        default_storage.delete(file_name)
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        return Response({'prediction': classes[int(predicted_class)], 'probability': prediction[0][predicted_class]}, status=status.HTTP_200_OK)
