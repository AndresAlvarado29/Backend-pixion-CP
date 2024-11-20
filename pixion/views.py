from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Image
from .serializers import ImageSerializer
from PIL import Image as PILImage
import io

class ProcessImageView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            # Guardar la imagen original
            instance = serializer.save()

            # aqui se llama  a los filtros
            original_path = instance.img.path
            img = PILImage.open(original_path).convert('L')  # Convertir a escala de grises

            # Guardar la imagen procesada
            processed_path = original_path.replace('original_images', 'processed_images')
            img.save(processed_path)
            instance.processed_img.name = processed_path.split('media/')[1]  # Ajustar el path relativo
            instance.save()

            return Response(ImageSerializer(instance).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

