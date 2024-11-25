from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.generics import ListCreateAPIView, RetrieveAPIView
from rest_framework import status
from rest_framework.decorators import api_view
from .models import Image, Usuario
from .serializers import ImageSerializer, UsuarioSerializer
from PIL import Image as PILImage
import io

#POST
class UsuarioListCreateView(ListCreateAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

# Endpoint para buscar un usuario por username
@api_view(['GET'])
def buscar_usuario(request, username):
    try:
        usuario = Usuario.objects.get(username=username)
        serializer = UsuarioSerializer(usuario)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Usuario.DoesNotExist:
        return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)




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

