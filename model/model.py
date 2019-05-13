from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, TimeDistributed, Concatenate, Add
from keras.optimizers import Adam
from keras.regularizers import l2
from model.layer import *
from model.loss import std_mae, std_rmse


def model_3DGCN(molecule_parameters, molecule_size, num_features, loss, outputs, *args, **kwargs):
    # Kipf adjacency, neighborhood mixing
    units_conv = molecule_parameters["units_conv"]
    units_dense = molecule_parameters["units_dense"]
    num_layers = molecule_parameters["num_layers"]
    pooling = molecule_parameters["pooling"]
    num_atoms = molecule_size

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    sc, vc = GraphEmbed()([atoms, dists])

    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation='relu')(sc)
        sc_v = GraphVToS(units_conv, activation='relu')([vc, dists])

        vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
        vc_v = GraphVToV(units_conv, activation='tanh')(vc)

        sc = GraphConvS(units_conv, pooling='sum', activation='relu')([sc_s, sc_v, adjms])
        vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

    sc, vc = GraphGather(pooling=pooling)([sc, vc])
    sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
    sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc_out)

    vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc)
    vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc_out)
    vc_out = Flatten()(vc_out)

    out = Concatenate(axis=-1)([sc_out, vc_out])

    out = Dense(outputs, activation='sigmoid', name='output')(out)
    model = Model(inputs=[atoms, adjms, dists], outputs=out)
    model.compile(optimizer=Adam(lr=0.001), loss=loss)

    return model


def bi3DGCN(molecule_parameters, molecule_size, target_parameters, target_size, num_features, loss, outputs, *args, **kwargs):
    target_parameters['num_atoms'] = target_size
    molecule_parameters['num_atoms'] = molecule_size

    def submodel(model_params):
        # Kipf adjacency, neighborhood mixing
        num_atoms = model_params["num_atoms"]
        units_conv = model_params["units_conv"]
        units_dense = model_params["units_dense"]
        num_layers = model_params["num_layers"]
        pooling = model_params["pooling"]
        name = model_params["name"]

        atoms = Input(name=name + '_atom_inputs', shape=(num_atoms, num_features))
        adjms = Input(name=name + '_adjm_inputs', shape=(num_atoms, num_atoms))
        dists = Input(name=name + '_coor_inputs', shape=(num_atoms, num_atoms, 3))

        sc, vc = GraphEmbed()([atoms, dists])

        for _ in range(num_layers):
            sc_s = GraphSToS(units_conv, activation='relu')(sc)
            sc_v = GraphVToS(units_conv, activation='relu')([vc, dists])

            vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
            vc_v = GraphVToV(units_conv, activation='tanh')(vc)

            sc = GraphConvS(units_conv, pooling='sum', activation='relu')([sc_s, sc_v, adjms])
            vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc, vc = GraphGather(pooling=pooling)([sc, vc])
        sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
        sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc_out)

        vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc)
        vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc_out)
        vc_out = Flatten()(vc_out)

        out = Concatenate(axis=-1)([sc_out, vc_out])
        return (atoms, adjms, dists), out

    target_in, target_out = submodel(target_parameters)
    molecule_in, molecule_out = submodel(molecule_parameters)

    out = Concatenate(axis=-1)([target_out, molecule_out])

    out = Dense(outputs, activation='sigmoid', name='output')(out)
    model = Model(inputs=target_in + molecule_in, outputs=out)
    model.compile(optimizer=Adam(lr=0.001), loss=loss)

    return model


def model_2DGCN(molecule_parameters, molecule_size, target_parameters, target_size, num_features, loss, outputs, *args, **kwargs):
    target_parameters['num_atoms'] = target_size
    molecule_parameters['num_atoms'] = molecule_size

    def submodel(model_params):
        # Kipf adjacency, neighborhood mixing
        num_atoms = model_params["num_atoms"]
        units_conv = model_params["units_conv"]
        units_dense = model_params["units_dense"]
        num_layers = model_params["num_layers"]
        pooling = model_params["pooling"]
        name = model_params["name"]

        atoms = Input(name=name + '_atom_inputs', shape=(num_atoms, num_features))
        adjms = Input(name=name + '_adjm_inputs', shape=(num_atoms, num_atoms))
        dists = Input(name=name + '_coor_inputs', shape=(num_atoms, num_atoms, 3))

        sc, vc = GraphEmbed()([atoms, dists])

#         for _ in range(num_layers):
#             sc_s = GraphSToS(units_conv, activation='relu')(sc)
#             sc_v = GraphVToS(units_conv, activation='relu')([vc, dists])

#             vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
#             vc_v = GraphVToV(units_conv, activation='tanh')(vc)

#             sc = GraphConvS(units_conv, pooling='sum', activation='relu')([sc_s, sc_v, adjms])
#             vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc, vc = GraphGather(pooling=pooling)([sc, vc])
        sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
#         sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc_out)

        vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc)
#         vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc_out)
        vc_out = Flatten()(vc_out)

        out = Concatenate(axis=-1)([sc_out, vc_out])
        return (atoms, adjms, dists), out

    target_in, target_out = submodel(target_parameters)
    molecule_in, molecule_out = submodel(molecule_parameters)

    out = Concatenate(axis=-1)([target_out, molecule_out])

    out = Dense(outputs, activation='sigmoid', name='output')(out)
    model = Model(inputs=target_in + molecule_in, outputs=out)
    model.compile(optimizer=Adam(lr=0.001), loss=loss)

    return model

