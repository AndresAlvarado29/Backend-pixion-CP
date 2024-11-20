from rest_framework import serializers
from .models import Image, Usuario

class UsuarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Usuario
        fields = ['id','username', 'email' ]




class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'img', 'processed_img', 'user', 'likes', 'comments']
        read_only_fields = ['processed_img', 'likes', 'comments']  # No se pueden modificar likes/comments en esta vista

