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


 POST ```http://ipserver:8000/api/usuarios ```

----
LOGIN

se hace un post con username y password a:


POST ```http://ipserver:8000/api/login ```

Devuelve codigo ```http 200 OK ```  si el login es correcto 

caso contrario devuelve ```http 400 BAD_REQUEST```

----
Obtener usuarios por username (username es unico)



GET ```http://ipserver:8000/api/usuarios/buscar/<id_user>```


----

Obtener todos los usuarios  (solo para pruebas)


GET ```http://ipserver:8000/api/usuarios```
