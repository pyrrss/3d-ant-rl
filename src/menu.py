import questionary

MODELS = [
            "PPO",
            "A2C",
            "TRPO",
            "RecurrentPPO",
        ]

def select_model_saved() -> tuple[bool, str | None, str | None]:
    name_model = None
    model_file = None

    model_saved = questionary.select("¿Desea cargar un modelo existente?: ", choices=["True", "False"]).ask()
    model_saved = True if model_saved == "True" else False
    
    name_model = questionary.select("Seleccione un modelo a entrenar: ", choices=MODELS).ask()

    if model_saved:        
        model_file= questionary.text("Ingrese el nombre del modelo: ").ask()

    return model_saved, name_model, model_file