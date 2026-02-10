"""
    models_params.py
    Author: Milan Marocchi

    Purpose: Contains the parameters for best results of certain models
"""

import logging
import math
from learners.training import TrainerArguments

def get_model_training_args_from_file(
        model_config_file: str,
        num_models: int,
        output_path: str,
        num_epochs: int,
        optim: str,
) -> list[TrainerArguments]:
    """
        Reads a model config file and returns the model training args as a list of TrainerArguments.
        The file should contain one JSON object per line.
    """
    import json
    with open(model_config_file, 'r') as f:
        config = json.load(f)
    configs = [config]
    
    return [
        TrainerArguments(
            output_dir=config.get("output_path", output_path),
            batch_size=config.get("batch_size", 64),
            num_epochs=num_epochs,
            optim=config.get("optim", optim),
            learing_rate=config.get("learning_rate", 1e-4),
            step_size=config.get("step_size", 5),
            mini_batch=config.get("mini_batch", 64),
            weight_decay=config.get("weight_decay", 0.0),
            momentum=config.get("momentum", 0.9),
            max_grad_norm=config.get("max_grad_norm", None)
        ) for config in configs
    ]

def get_model_config_from_file(
        model_config_file: str,
        num_inputs: int,
        num_models: int,
        num_classes: int,
        fs: int,
        fragment_time: int,
) -> list[dict]:
    """
        Reads a model config file and returns the model config as a list of dictionaries.
        The file should contain one JSON object per line.
    """
    import json
    with open(model_config_file, 'r') as f:
        config = json.load(f)
    configs = [config]
    # Inject/overwrite params into each config dict
    for config in configs:
        config["num_channels"] = num_inputs
        config["num_models"] = num_models
        config["num_classes"] = num_classes
        config["fs"] = fs
        config["fragment_time"] = fragment_time
    return configs

def get_model_config(
        model_code: str, 
        num_inputs: int, 
        num_models: int, 
        dataset: str, 
        num_classes: int, 
        fs: int,
        fragment_time: int,
        large_model: bool = False,
        is_ecg: bool = False,
) -> list[dict]:
    """
        Contains the optimal configs for various types of models 
    """
    if model_code == "mfcconformer":
        return [
            {
                "num_classes": num_classes,
                "mlp_hidden_dim": 512,
                "hidden_dim": 1024,
                "n_mels": 80,
                "input_dim": 48,
                "num_heads": 4,
                "num_blocks": 4,
                "num_layers": 3,
                "alpha": 0.5828256081181153,
                "beta": 0.7044231098956046,
                "dropout": 0.28094364608354827,
                "center": 0.0005596328065221168,
                "temperature": 0.833051981452277,
                "num_channels": num_inputs,
                "fs": fs,
                "fragment_time": fragment_time,
                "max_grad_norm": 1.0,
            } for _ in range(num_models + 1 if large_model else num_models)
        ]
    if model_code == "mfcccnn":
        return [
            {
                "num_classes": num_classes,
                "hidden_layer_sizes": [2048, 1024, 512],
                #"cnn_features": [32, 32],
                #"cnn_strides": [2, 2],
                #"cnn_filters": [2, 2],
                "dropout": 0.2,
                "frame_length": 0.036,
                "lambda_c": 0.0,
                "alpha": 1.0,
                "beta": 0.0,
                #"temperature": 0.2989053333748626,
                "num_in_channels": num_inputs,
                "fs": fs,
                "fragment_time": fragment_time
            } for _ in range(num_models + 1 if large_model else num_models)
        ]
    if model_code == "vae":
        if dataset == "training-a":
            return [
                {
                    "num_classes": num_classes,
                    "in_channels": 1
                },
                {
                    "num_classes": num_classes,
                    "in_channels": 1
                },
                {
                    "num_classes": num_classes,
                    "hidden_size": [512, 512, 512],
                    "num_inputs": num_inputs
                },
            ]
    if model_code == "wav2vecss":
        return [
            {
                "num_classes": num_classes,
            }]
    if model_code == "wav2vec":
        if dataset == "cinc":
            return [
                {
                    "num_classes": num_classes,
                    "hidden_size": [512, 512, 512]
                }
            ]
        if dataset == "training_a":
            return [
                {
                    "num_classes": num_classes,
                    "hidden_size": 512,
                }
            ]
        if dataset == "training_a-e":
            return [
                {
                    "num_classes": num_classes,
                    "hidden_size": 128,
                }
            ]
        if num_inputs == 2 and dataset == "training-a":
            return [
                {
                    "num_classes": num_classes,
                    "hidden_size": 512,
                },
                {
                    "num_classes": num_classes,
                    "hidden_size": 128,
                },
                {
                    "num_classes": num_classes,
                    "hidden_layer_sizes": [1024, 1024, 1024] # [1024, 1024]
                }
            ]
    if model_code in ["parakeet", "wav2vec"]:
        config = [
            {
                "num_models": num_models, 
                "fs": fs, 
                "fragment_time": fragment_time,
                "lambda_c": 0.6035278609026304,
                "alpha": 0.6020565941781555,
                "beta": 0.9771014704583595,
                "temperature": 0.3091871085194138,
                "num_channels": num_inputs,
            } for i in range(int(num_models) + 1 if large_model else int(num_models))]
    if dataset in ['ticking-heart', 'vest-data'] and model_code == "whisper":
        config = [
            {
                "num_models": num_models, 
                "num_channels": num_inputs, 
                "fs": fs, 
                "fragment_time": fragment_time,
                "is_ecg": i == num_models-1, # Change spectrogram params for ecg
                "hidden_layer_sizers": [1024, 1024],
                "lambda_c": 0.6035278609026304,
                "alpha": 0.6020565941781555,
                "beta": 0.9771014704583595,
                "temperature": 0.3091871085194138,
                "max_grad_norm": 2.0,
            } for i in range(int(num_models) + 1 if large_model else int(num_models))]
    if dataset in ['ticking-heart', 'vest-data'] and model_code in ["wav2vec_cnn", "wav2conformer"]:
        if large_model:
            config: list[dict] = [
                {
                    "num_classes": num_classes,
                    "hidden_size": 512,
                    "num_channels": num_inputs,
                }
            for _ in range(int(num_models))]
            config.append({
                "num_classes": num_classes,
                "hidden_layer_sizes": [512],
                "num_channels": num_inputs,
                "num_models": int(num_models),
                "lambda_c": 0.7394646239481701,
                "alpha": 0.06475544231030396,
                "beta": 0.6968610616487353,
                "temperature": 0.31762545902840156
            })
        else:
            config: list[dict] = [
                {
                    "num_classes": num_classes,
                    "hidden_size": [512],
                    "num_channels": num_inputs,
                    "lambda_c": 0.7394646239481701,
                    "alpha": 0.06475544231030396,
                    "beta": 0.6968610616487353,
                    "temperature": 0.31762545902840156
                }]

        return config
    if dataset in ["ticking-heart", "vest-data"] and model_code == "wav2vec":
        if large_model:
            config: list[dict] = [
                {
                    "num_classes": num_classes,
                    "hidden_size": 512,
                    "num_channels": num_inputs,
                }
            for _ in range(int(num_models))]
            config.append({
                "num_classes": num_classes,
                "hidden_layer_sizes": [2048, 2048, 1024],
                "num_models": int(num_models),
            })
        else:
            config: list[dict] = [
                {
                    "num_classes": num_classes,
                    "hidden_size": [1024, 1024],
                    "num_channels": num_inputs,
                }]

        return config
    elif model_code == "cnn":
        return [
            {
                "num_classes": num_classes,
                "hidden_size": 1024,
                "num_in_channels": num_inputs
            }
        for _ in range(num_inputs)]
    elif model_code == "heartformer":
        return [{
                "num_classes": num_classes,
                "hidden_sizes": [256, 256],
                "num_in_channels": num_inputs
        } for _ in range(int(num_models) + 1 if large_model else int(num_models))]
    elif model_code == "mamba":
        output = [{
                "num_classes": num_classes,
                "hidden_sizes": [256, 256],
                "num_in_channels": num_inputs
        } for _ in range(int(num_models))]

        if large_model:
            output.append({
                "num_classes": num_classes,
                "num_channels": num_inputs,
                "hidden_layer_sizes": [1024, 1024],
            })

        return output

    # Otherwise return default
    return [{"num_models": num_models, "num_channels": num_inputs, "fs": fs, "fragment_time": fragment_time} for _ in range(int(num_models) + 1 if large_model else int(num_models))]


def get_model_training_args(
        model_code: str, 
        num_inputs: int, 
        num_models: int,
        dataset: str,
        output_path: str,
        num_epochs: int,
        optim: str,
        large_model: bool = False,
        lora: bool = True
) -> list[TrainerArguments]:

    if model_code == "mfcccnn":
        return [
            TrainerArguments(
                output_path, 
                batch_size=128,
                num_epochs=num_epochs,
                step_size=4,
                gamma=0.10084233554259762,
                optim="adamw",
                learing_rate=1e-4,
                momentum=0.7,
                weight_decay=5.260679690558362e-05
            ) for _ in range(num_models + 1 if large_model else num_models)
        ]
    if model_code == "vae":
        if dataset == "training-a":
            return [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=64, #8
                    num_epochs=num_epochs,
                ),
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=64,
                    num_epochs=num_epochs,
                ),
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=16,
                    num_epochs=num_epochs,
                ),
            ]
    if "whisper" in model_code:
        if not large_model:
            return [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=64,
                    num_epochs=num_epochs,
                    optim='adamw',
                    learing_rate=1e-4,
                    weight_decay=1e-4,
                    step_size=5,
                    mini_batch=64
                )
            ]
        else:
            batch_size = 256 if model_code == "whisper" else 128
            args = [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    optim='adamw',
                    learing_rate=1e-4,
                    step_size=1,
                    mini_batch=64
                ) for _ in range(num_models)]
            args.append(
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    optim='adam',
                    step_size=8,
                    mini_batch=32,
                    learning_rate=7.410652815776028e-05,
                    momentum=0.5307740049911385,
                    weight_decay=7.599571446717488e-05,
                    gamma=0.010306725447514313,
                )
            )
            return args
    if "parakeet" in model_code:
        return [TrainerArguments(
                        output_dir=output_path,
                        batch_size=256,
                        num_epochs=num_epochs,
                        optim='adam',
                        learing_rate=1e-5,
                        step_size=10
        )]
    if "ast" in model_code:
        if model_code == "ast":
            if not large_model:
                return [
                    TrainerArguments(
                        output_dir=output_path,
                        batch_size=64,
                        num_epochs=num_epochs,
                        optim='adamw',
                        learing_rate=1e-5,
                        step_size=1,
                        mini_batch=64
                    )
                ]
            else:
                args = [
                    TrainerArguments(
                        output_dir=output_path,
                        batch_size=32,
                        num_epochs=num_epochs,
                        optim='adamw',
                        learing_rate=1e-5,
                        step_size=10
                    )
                for _ in range(num_models)]
                args.append(
                    TrainerArguments(
                        output_dir=output_path,
                        batch_size=32,
                        num_epochs=num_epochs,
                        optim='adamw',
                        learing_rate=1e-5,
                        step_size=10
                    )
                )
                return args
        else:
            return [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=128,
                    mini_batch=48,
                    num_epochs=num_epochs,
                    optim='adamw',
                    learing_rate=1e-5,
                    step_size=10
                )
            ]

    if model_code in ["mfccmamba", "mfcconformer", "wav2vec"]:
        args = [
            TrainerArguments(
                output_path,
                batch_size=256,
                mini_batch=64,
                num_epochs=num_epochs,
                optim="adamw",
                num_channels=num_inputs,
                learning_rate=0.00018416367446326122,
                step_size=9,
                momentum=0.9607886767616296,
                weight_decay=2.953834843634957e-05,
                gamma=0.275950598987826,
                max_grad_norm=1.0,
            ) for _ in range(num_models)]
        args.append(
            TrainerArguments(
                output_path,
                batch_size=256,
                mini_batch=64,
                num_epochs=num_epochs,
                optim="adamw",
                num_channels=num_inputs,
                learning_rate=0.00018416367446326122,
                step_size=9,
                momentum=0.9607886767616296,
                weight_decay=2.953834843634957e-05,
                gamma=0.275950598987826,
                max_grad_norm=1.0,
            )
        )
        return args

    if "wav2vec" in model_code:
        if not large_model:
            if model_code == "wav2vecss" or lora and model_code == "wav2vec":
                    args = [
                        TrainerArguments(
                            output_path, 
                            batch_size=128,
                            num_epochs=num_epochs,
                            optim="adamw",
                            mini_batch=128,
                            step_size=20,
                            max_grad_norm=5.0
                        )]
                    return args
        else:
            if model_code in ["wav2vec_cnn", "wav2conformer"]:
                args = [
                    TrainerArguments(
                        output_path, 
                        batch_size=128,
                        num_epochs=num_epochs,
                        optim="adam",
                        mini_batch=128,
                        learning_rate=1e-4,
                        step_size=2,
                    ) for _ in range(num_models)]
                args.append(
                    TrainerArguments(
                        output_path, 
                        batch_size=16,
                        num_epochs=num_epochs,
                        optim="adam",
                        mini_batch=16,
                        learing_rate=0.0019994343994189563,
                        momentum=0.5829911553445423,
                        weight_decay=7.250913282675207e-05,
                        gamma=0.026955692251800664,
                        step_size=5,
                    )
                )
                return args
            if model_code == "wav2vecss" or lora and model_code == "wav2vec":
                    args = [
                        TrainerArguments(
                            output_path, 
                            batch_size=64,
                            num_epochs=num_epochs,
                            optim="adamw",
                            mini_batch=64,
                            step_size=20,
                            max_grad_norm=5.0
                        ) for _ in range(num_models)]
                    args.append(
                        TrainerArguments(
                            output_path, 
                            batch_size=16,
                            num_epochs=num_epochs,
                            optim="rmsprop",
                            mini_batch=16
                        )
                    )
                    return args
        if dataset == "vest-data":
            if not large_model:
                if model_code != "wav2vec":
                    return [
                        TrainerArguments(
                            output_path, 
                            batch_size=32,
                            num_epochs=num_epochs,
                            learning_rate=1e-5,
                            weight_decay=6.1148e-05,
                            momentum=0.17562,
                            step_size=4,
                            gamma = 0.02444,
                            optim="rmsprop",
                            mini_batch=32
                        )
                    ]
                else:
                    return [
                        TrainerArguments(
                            output_path,
                            batch_size=64,
                            num_epochs=num_epochs,
                            optim="sgd",
                            mini_batch=32
                        )
                    ]
            else:
                args = [
                    TrainerArguments(
                        output_path, 
                        batch_size=32,
                        num_epochs=num_epochs,
                        learning_rate=1e-5,
                        weight_decay=6.1148e-05,
                        momentum=0.17562,
                        step_size=4,
                        gamma = 0.02444,
                        optim="rmsprop",
                        mini_batch=32
                    )
                for _ in range(num_models)]
                args.append(
                    TrainerArguments(
                        output_path, 
                        batch_size=32,
                        num_epochs=num_epochs,
                        learning_rate=1e-5,
                        weight_decay=6.1148e-05,
                        momentum=0.17562,
                        step_size=4,
                        gamma = 0.02444,
                        optim="rmsprop",
                        mini_batch=32
                    )
                )
                return args
        elif dataset == "cinc":
            return [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=64, #64
                    num_epochs=num_epochs,
                    learning_rate= 0.001,#0.05660669917352466,#0.02077354046195862,
                    weight_decay= 4.11e-5,#2.009286725841079e-05,#1.745662930118604e-05,
                    momentum= 0.57562,#0.5787515844599177,#0.6727905252512073,
                    step_size= 2,#1,#1,
                    gamma = 0.167,#0.32819675163508516,#0.16725037645605495,
                    optim="sgd"
                )
            ]
        if dataset == "training_a":
            return [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=64, #64
                    num_epochs=num_epochs,
                    learning_rate= 0.001,#0.05660669917352466,#0.02077354046195862,
                    weight_decay= 3.11e-5,#2.009286725841079e-05,#1.745662930118604e-05,
                    momentum= 0.17562,#0.5787515844599177,#0.6727905252512073,
                    step_size= 7,#1,#1,
                    gamma = 0.002444,#0.32819675163508516,#0.16725037645605495,
                    optim="sgd"
                )
            ]
        elif dataset == "training_a-e":
            return [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=32, #64
                    num_epochs=num_epochs,
                    learning_rate= 0.015749,#0.05660669917352466,#0.02077354046195862,
                    weight_decay= 4.89874e-05,#2.009286725841079e-05,#1.745662930118604e-05,
                    momentum= 0.4342338,#0.5787515844599177,#0.6727905252512073,
                    step_size= 2,#1,#1,
                    gamma = 0.489874,#0.32819675163508516,#0.16725037645605495,
                    optim="sgd"
                )
            ]
        # THIS IS THE PARAMS FROM AN OLD OPTIM THAT GAVE BEST RESULTS
        elif num_inputs == 2 and dataset == "training-a":
            return [
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=64, #8
                    num_epochs=num_epochs,
                    learning_rate=0.0010967,
                    weight_decay=3.1148e-05,
                    momentum=0.17562,
                    step_size=7,
                    gamma = 0.002444,
                    optim="sgd"
                ),
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=32,
                    num_epochs=num_epochs,
                    learning_rate=0.015749,
                    weight_decay=4.89874e-05,
                    momentum=0.4342338,
                    step_size=2,
                    gamma = 0.489874,
                    optim="sgd"
                ),
                TrainerArguments(
                    output_dir=output_path,
                    batch_size=16,
                    num_epochs=num_epochs,
                    learning_rate=2.0458e-05,
                    weight_decay=7.36e-05,
                    momentum=0.5447,
                    step_size=9,
                    gamma = 0.05825,
                    optim="sgd"
                ),
            ]
        if dataset == "ticking-heart" or dataset == "vest-data":
            args = [
                TrainerArguments(
                    output_dir=f"{output_path}_model_part_{i}",
                    batch_size=64,
                    num_epochs=num_epochs,
                    learning_rate=0.0016552,
                    weight_decay=3.547e-05,
                    momentum=0.1761,
                    step_size=7,
                    gamma = 0.04588,
                    optim="sgd"
                )
            for i in range(num_inputs)]
            args.append(TrainerArguments(
                output_dir=output_path,
                batch_size=32,
                num_epochs=num_epochs,
                learning_rate=0.001828,
                weight_decay=3.8e-07,
                momentum=0.512,
                step_size=4,
                gamma = 0.0763,
                optim="sgd"
            ))
            return args
    elif model_code == "cnn":
        return [
            TrainerArguments(
                output_path, 
                batch_size=128,
                num_epochs=num_epochs, 
                optim=optim
            ) for _ in range(num_inputs + 1 if large_model else num_inputs)
        ]
    elif model_code == "heartformer":
        return [
            TrainerArguments(
                output_path, 
                batch_size=64, #8
                num_epochs=num_epochs,
                learning_rate=1e-4,
                weight_decay=3.1148e-05,
                momentum=0.17562,
                step_size=15,
                gamma = 0.002444,
                optim="rmsprop"
            ) 

            for _ in range(int(num_models) + 1 if large_model else int(num_models))
       ] 
    elif "mamba" in model_code:
        output = [
            TrainerArguments(
                output_path, 
                batch_size=64,
                num_epochs=num_epochs,
                learning_rate=1e-4,
                weight_decay=6.1148e-05,
                momentum=0.17562,
                step_size=2,
                gamma = 0.02444,
                optim="rmsprop",
                mini_batch=64
            ) 
            for _ in range(int(num_models))
       ] 

        if large_model:
            output.append(
                TrainerArguments(
                    output_path, 
                    batch_size=128,
                    num_epochs=num_epochs,
                    learning_rate=1e-4,
                    weight_decay=3.1148e-05,
                    momentum=0.17562,
                    step_size=5,
                    gamma = 0.02444,
                    optim="rmsprop",
                    mini_batch=128
            )
        )
        return output
    if dataset == "vest-data" and large_model:
        output = [
            TrainerArguments(
                f"{output_path}_part_{i}", 
                num_epochs=num_epochs, optim=optim) for i in range(int(num_models))
        ]
        output.append(
            TrainerArguments(output_path, num_epochs=math.ceil(num_epochs))
        )
        return output

    # Otherwise return default
    return [TrainerArguments(f"{output_path}_part_{i}", num_epochs=num_epochs, optim=optim) for i in range(int(num_models) + 1 if large_model else int(num_models))]
