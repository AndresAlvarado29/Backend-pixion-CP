from rest_framework import serializers
from .models import Image, Usuario, Comment
from django.contrib.auth.hashers import make_password


# aqui se define como se muestra el JSON de respuesta ya sea GET POST, ... etc
# serializer para el registro
class UsuarioSerializerRegistro(serializers.ModelSerializer):
    class Meta:
        model = Usuario
        fields = ['id','username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    # este metodo omite el campo de la contrasenia solo al momento de recibir (GET), por seguridad.
    # se sobreescribe este metodo (overwrite)
    def to_representation(self, instance):
        representation = super().to_representation(instance)
        representation.pop('password', None)
        return representation

# serializer general
class UsuarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Usuario
        fields = ['id', 'username', 'email']  # Excluye el campo password

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        return representation


# serializaer para el login
class UsuarioLoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ['id', 'id_user','date','comment']

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'img_original', 'img_processed', 'id_user', 'likes', 'comments']
        read_only_fields = ['img_processed', 'likes']  # No se pueden modificar likes/comments en esta vista


class ImageListSerializer(serializers.ModelSerializer):
    comments = CommentSerializer(many=True, read_only=True)
    class Meta:
        model = Image
        fields = ['id', 'img_processed', 'id_user', 'likes', 'comments']

