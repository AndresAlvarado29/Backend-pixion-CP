from django.db import models
from django.contrib.auth.hashers import make_password


class Usuario(models.Model):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # Se almacena el hash de la contraseña

    def save(self, *args, **kwargs):
        # se cifra la contrasenia y se guarda el hash
        if not self.pk:  # solo para usuarios nuevos
            self.password = make_password(self.password)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.username


class Image(models.Model):
    id_user = models.ForeignKey(Usuario, on_delete=models.CASCADE, related_name="images")
    img_original = models.ImageField(upload_to="images/originals/")
    img_processed = models.ImageField(upload_to="images/processed/", blank=True, null=True)
    likes = models.PositiveIntegerField(default=0)
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"Image {self.id} by {self.id_user.username}"


class Comment(models.Model):
    id_image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name="comments")
    id_user = models.ForeignKey(Usuario, on_delete=models.CASCADE, related_name="comments")
    date = models.DateTimeField(auto_now_add=True)
    comment = models.TextField()
    # ordena los comentarios por fecha
    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"Comment by {self.id_user.username} on Image {self.id_image.id}"
