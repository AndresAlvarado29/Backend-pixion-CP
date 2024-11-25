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

### Endpoints (APIs)

creacion de usuarios


 POST ```http://ipserver:8000/api/usuarios ```

----

Obtener usuarios por username (username es unico)



GET ```http://ipserver:8000/api/usuarios/buscar/<id_user>```


----

Obtener todos los usuarios  (solo para pruebas)


GET ```http://ipserver:8000/api/usuarios```
