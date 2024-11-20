from django.db import models



class Usuario(models.Model):
    username = models.CharField(max_length=50, unique=True)
    # guardar el hash de la passwd
    passwd = models.CharField(max_length=256)  

    email = models.EmailField(unique=True)

    def __str__(self):
        return self.username


class Image(models.Model):
    img = models.ImageField(upload_to='original_images/')  # Imagen original
    processed_img = models.ImageField(upload_to='processed_images/', null=True, blank=True)  # Imagen procesada
    user = models.ForeignKey('Usuario', on_delete=models.CASCADE, related_name='images')
    likes = models.PositiveIntegerField(default=0)
    comments = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f'Image {self.id} - User: {self.user.username}'
