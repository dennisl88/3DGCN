from model.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer("aa2ar")

    target_parameters = {"units_conv": 128, "units_dense": 128, "pooling": "max", "num_layers": 2,}
    molecule_parameters = {"units_conv": 128, "units_dense": 128, "pooling": "max", "num_layers": 2,}

    hyperparameters = {"epoch": 150, "batch": 16, "fold": 10,  "loss": "binary_crossentropy", "monitor":
                       "val_roc", "label": "", "target_paramters": target_parameters, "molecule_paramters": molecule_parameters}

    features = {"use_atom_symbol": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_ring_size": True, "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}

    # Baseline
    trainer.fit('bi3DGCN', **hyperparameters, **features)