from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, TimeDistributed, Concatenate, Add
from keras.optimizers import Adam
from keras.regularizers import l2
from model.layer import *
from model.loss import std_mae, std_rmse


def model_3DGCN(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    loss = hyper["loss"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]

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


def bi3DGCN(hyper):
    outputs = hyper["outputs"]
    loss = hyper["loss"]
    hyper['target_model']['num_atoms'] = hyper['target_size']
    hyper['molecule_model']['num_atoms'] = hyper['molecule_size']

    def submodel(model_params):
        # Kipf adjacency, neighborhood mixing
        num_atoms = model_params["num_atoms"]
        num_features = hyper["num_features"]
        units_conv = model_params["units_conv"]
        units_dense = model_params["units_dense"]
        num_layers = model_params["num_layers"]
        pooling = model_params["pooling"]

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
        return (atoms, adjms, dists), out

    target_in, target_out = submodel(hyper['target_model'])
    molecule_in, molecule_out = submodel(hyper['molecule_model'])

    out = Concatenate(axis=-1)([target_out, molecule_out])

    out = Dense(outputs, activation='sigmoid', name='output')(out)
    model = Model(inputs=target_in + molecule_in, outputs=out)
    model.compile(optimizer=Adam(lr=0.001), loss=loss)

    return model
