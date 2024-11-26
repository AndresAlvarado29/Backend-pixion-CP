from django.urls import path
from .views import ProcessImageView, UsuarioListCreateView, buscar_usuario, LoginView, ImageListView, UserImagesView, \
    BuscarUsuarioByIdView

urlpatterns = [
    path('process-image/', ProcessImageView.as_view(), name='process-image'),
    path('usuarios/', UsuarioListCreateView.as_view(), name='listar_crear_usuarios'),
    path('usuarios/buscar/<str:username>/', buscar_usuario, name='buscar_usuario'),
    path('usuarios/buscarid/<int:id_user>/', BuscarUsuarioByIdView.as_view(), name='buscar-usuario-by-id'),
    path('login/', LoginView.as_view(), name='login'),
    path('images/', ImageListView.as_view(), name='images-list'),
    path('images/user/<int:id_user>/', UserImagesView.as_view(), name='user-images'),

]
