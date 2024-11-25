from django.contrib.auth.hashers import check_password
from django.core.serializers import serialize
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.generics import ListCreateAPIView, RetrieveAPIView
from rest_framework import status
from rest_framework.decorators import api_view
from .models import Image, Usuario
from .serializers import ImageSerializer, UsuarioSerializer, UsuarioLoginSerializer, UsuarioSerializerRegistro
from PIL import Image as PILImage
import io

#POST
class UsuarioListCreateView(ListCreateAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializerRegistro

# Endpoint para buscar un usuario por username (pruebas)
@api_view(['GET'])
def buscar_usuario(request, username):
    try:
        usuario = Usuario.objects.get(username=username)
        serializer = UsuarioSerializer(usuario)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Usuario.DoesNotExist:
        return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)

#endpoint para el login, buscar usuario y verificar contrasenia
class LoginView(APIView):
    def post(self, request):
        serializer = UsuarioLoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        username = serializer.validated_data['username']
        password = serializer.validated_data['password']


        # verificar si existe el usuario
        try:
            usuario = Usuario.objects.get(username=username)
        except Usuario.DoesNotExist:
            return Response({"Error: ": "Usuario no existe"}, status=status.HTTP_404_NOT_FOUND)

        # si existe verifica la contrasenia

        if not check_password(password, usuario.password):
            return Response({"Error": "Contraseña incorrecta"}, status=status.HTTP_400_BAD_REQUEST)

        serializer = UsuarioSerializer(usuario)
        return Response(serializer.data, status=status.HTTP_200_OK)

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

