from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PredictView(APIView):
    def post(self, request):
        data = request.data
        print(data)
        prediction = data
        return Response({'prediction': prediction}, status=status.HTTP_200_OK)
