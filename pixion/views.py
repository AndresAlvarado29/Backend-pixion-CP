from django.contrib.auth.hashers import check_password
from django.core.serializers import serialize
from django.shortcuts import render
from pycuda.curandom import random_source
from rest_framework.views import APIView
from rest_framework.exceptions import NotFound
from rest_framework.response import Response
from rest_framework.generics import ListCreateAPIView, RetrieveAPIView, ListAPIView
from rest_framework import status
from rest_framework.decorators import api_view
from .models import Image, Usuario
from .serializers import ImageSerializer, ImageListSerializer , UsuarioSerializer, UsuarioLoginSerializer, UsuarioSerializerRegistro
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


# endpoint para buscar usuario por id
class BuscarUsuarioByIdView(RetrieveAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

    def get_object(self):
        try:
            return Usuario.objects.get(id=self.kwargs['id_user'])
        except Usuario.DoesNotExist:
            raise NotFound('Usuario no encontrado')


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

# endpoint para aplicar los filtros
class ProcessImageView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            # Guardar la imagen original
            instance = serializer.save()

            # aqui se llama  a los filtros
            original_path = instance.img_original.path
            img = PILImage.open(original_path).convert('L')  # Convertir a escala de grises

            # Guardar la imagen procesada
            processed_path = original_path.replace('original_images', 'processed_images')
            img.save(processed_path)
            instance.img_processed.name = processed_path.split('media/')[1]  # Ajustar el path relativo
            instance.save()

            return Response(ImageSerializer(instance).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# endpoint para mostrar las images (feed)
class ImageListView(ListAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageListSerializer

# endpoint para mostrar las imagenes del usario
class UserImagesView(ListAPIView):
    serializer_class = ImageListSerializer

    def get_queryset(self):
        id_user = self.kwargs['id_user']
        return Image.objects.filter(id_user=id_user).order_by('-date')