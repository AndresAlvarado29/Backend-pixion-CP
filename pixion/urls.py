from django.urls import path
from .views import ProcessImageView, UsuarioListCreateView, buscar_usuario, LoginView

urlpatterns = [
    path('process-image/', ProcessImageView.as_view(), name='process-image'),
    path('usuarios/', UsuarioListCreateView.as_view(), name='listar_crear_usuarios'),
    path('usuarios/buscar/<str:username>/', buscar_usuario, name='buscar_usuario'),
    path('login/', LoginView.as_view(), name='login'),
]
