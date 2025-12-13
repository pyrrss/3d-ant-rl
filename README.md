Este proyecto entrena y evalúa distintos modelos de Deep Reinforcement Learning para desempeñarse dentro del
entorno Ant-V5 de Gymnasium.


# Instalación de dependencias
Las principales dependencias son Gymnasium (https://gymnasium.farama.org/) y Stable-Baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

Para instalar todas las dependencias necesarias, ejecutar:

```bash
pip install -r requirements.txt
```

# Entrenamiento de modelos

El proyecto soporta los siguientes modelos de DeepRL: A2C, TRPO, PPO y RecurrentPPO.

Para realizar el entrenamiento de un modelo, desde el directorio raíz:

```bash
python main.py --model <model>
```
Donde model = {A2C, TRPO, PPO, RecurrentPPO}

Durante el entrenamiento se va construyendo un archivo logs/*_monitor.csv (dependiendo del modelo entrenado) con datos de entrenamiento en tiempo real.
Cabe destacar que en el directorio raíz están disponibles modelos ya entrenados (archivos .zip) con 5.000.000 steps.

# Evaluación de modelos

Para evaluar un modelo ya entrenado, este se debe encontrar el directorio raíz del proyecto (en formato .zip) y ejecutar:

```
python main.py --load True --model <model> --file <model>_Ant.zip
```

A partir de la evaluación se genera un archivo logs/avg_rewards.csv con datos de la evaluación (para la evaluación de cada modelo
los datos se concatenan en el mismo csv).


# Uso de herramientas disponibles

## Visualización y guardado de métricas

Para visualizar y guardar gráficos de las curvas de aprendizaje de los modelos y sus recompensas medias, ejecutar 
desde el directorio raíz:

```bash
python -m src.tools.plots
```

Para el funcionamiento correcto de este script debe estar presente el o los archivos logs/*_monitor.csv para la construcción de las curvas
de aprendizaje (se toman todos los disponibles) y logs/avg_rewards.csv para la contrucción del gráfico de recompensas medias.


## Grabación y guardado de videos

Para realizar y guardar grabaciones de los modelos, ejecutar:

```bash
python -m src.tools.record_videos --model <model>
```
Donde <model> = {A2C, TRPO, PPO, RecurrentPPO}

El propósito principal de esta herramienta es realizar grabaciones de los modelos en distintas etapas de aprendizaje. Actualmente la configuración
se realiza dentro del script (se debe mejorar CLI). Durante el entrenamiento se guardan checkpoints del modelo cada 100.000 steps (se guardan en checkpoints/),
estos actualmente se cargan en el script y se realiza una evaluación grabada del modelo en ese estado para posteriormente guardarla en videos/.


# Equipo

| Name | Github |
|------|--------|
|Juan Felipe Raysz Muñoz|[@Sephir0ath](https://github.com/Sephir0ath)|
|Francisca Isidora Núñez Larenas|[@sshiro0](https://github.com/sshiro0)|
|Javier Alejandro Campos Contreras|[@4lehh](https://github.com/4lehh)|
|Oliver Isaías Peñailillo Sanzana|[@pyrrss](https://github.com/pyrrss)|

