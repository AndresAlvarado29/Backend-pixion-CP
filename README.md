## Proyecto Interciclo Computacion Paralela P65
### Aplicacion de filtros usando PyCuda
### Integrantes
* Alvarado Nixon
* Astudillo Paul
* Tapia Diego

Proyecto desarrollado en Django
Base de datos a usar SQLite
Se va "Dockerizar" con la imagen de Ubuntu22:Cuda12.4

Las librerias a usar estan en el archivo ```requirements.txt```

### Ejecucion

al ejecutar el servidor por primera vez se debe ejecutar los siguientes comandos para migrar la base de datos:

```python manage.py makemigrations```

```python manage.py migrate```

Luego se puede ejecutar el servidor (se ejecuta en el puerto 8000)

```python manage.py runserver```

### Endpoints (APIs)

creacion de usuarios


 POST ```http://<ip-server>:8000/api/usuarios ```

----
LOGIN

se hace un post con username y password a:


POST ```http://<ip-server>:8000/api/login ```

Devuelve codigo ```http 200 OK ```  si el login es correcto 

caso contrario devuelve ```http 400 BAD_REQUEST```

----
Obtener usuarios por username (username es unico)


GET  ```http://ipserver:8000/api/usuarios/buscar/<username>```



----
Obtener usuarios por id

GET  ```http://ipserver:8000/api/usuarios/buscarid/<id-user>```


----

Obtener todas la imagenes (feed)

GET ```http://<ip-server>:8000/api/images```

las imagenes estan ordenadas por fecha de creacion de manera descendente 

----

Obtener las imagenes del usuario

GET ```http://ipserver:8000/api/images/user/<id_user>```

----

Obtener todos los usuarios  (solo para pruebas)


GET ```http://ipserver:8000/api/usuarios```

----

Endpoint para dar like 

POST ```http://ipserver:8000/api/images/<id_image>/like  id_user=<id_user>```

ejemplo:

POST al endpoint con el siguiente json
```{"id_user"=2}```

----
Endpoint para comentar 

POST ```http://ipserver:8000/api/add-comment/```

El endpoint /add-comment/ acepta un POST con los siguientes datos:
```id_image:``` El id de la imagen a la que se le quiere agregar el comentario.
```id_user:``` El id del usuario que esta comentando.
```comment:``` El texto del comentario.

ejemplo:
```
{

    "id_image"=1,

    "id_user"=1,

    "comment"="Este es un comentario en la imagen."
}
```


----

Endpoint para aplicar filtros 
POST ```http://ipserver:8000/api/process-image/```

{ "id_user"=1, "filter_type": "gauss" }


----