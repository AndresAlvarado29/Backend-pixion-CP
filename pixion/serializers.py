from rest_framework import serializers
from .models import Image, Usuario
from django.contrib.auth.hashers import make_password


# aqui se define como se muestra el JSON de respuesta ya sea GET POST, ... etc

class UsuarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Usuario
        fields = ['id','username', 'email', 'password']

    # se cifra la contrasenia y se guarda el hash
    def validate_password(self, value):
        return make_password(value)
    
    # este metodo omite el campo de la contrasenia solo al momento de recibir (GET), por seguridad.
    # se sobreescribe este metodo (overwrite)
    def to_representation(self, instance):
        representation = super().to_representation(instance)
        representation.pop('password')
        return representation
        


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'img', 'processed_img', 'user', 'likes', 'comments']
        read_only_fields = ['processed_img', 'likes', 'comments']  # No se pueden modificar likes/comments en esta vista

