from keras.utils import to_categorical, Sequence
from rdkit import Chem
from rdkit.Chem import rdmolops, rdmolfiles, AllChem
import numpy as np
import os
import pickle as pkl
from joblib import Parallel, delayed

def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


class Dataset(object):
    def __init__(self, dataset=None, batch=128):
        self.dataset = dataset
        if dataset == None:
            self.path = "../../data/"
        else:
            self.path = "../../data/{0}/".format(dataset)
        self.task = "binary"
        self.target_name = "active"
        self.target_size = 0
        self.molecule_size = 0

        self.batch = batch
        self.outputs = 1

        self.mols = []
        self.coords = []
        self.target = []
        self.tx, self.tc, self.x, self.c, self.y = {}, {}, {}, {}, {}

        self.use_atom_symbol = True
        self.use_degree = True
        self.use_hybridization = True
        self.use_implicit_valence = True
        self.use_partial_charge = False
        self.use_formal_charge = True
        self.use_ring_size = True
        self.use_hydrogen_bonding = True
        self.use_acid_base = True
        self.use_aromaticity = True
        self.use_chirality = True
        self.use_num_hydrogen = True

        # Load data
        try:
            self.reload_dataset(self.path + 'dataset.pkl')
        except:
            self.load_dataset()

        # Calculate number of features
        mp = MPGenerator([], [], [], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()

    def load_dataset(self):
        # Load files
        x, c, y = [], [], []

        if self.dataset is None:
            tx, tc = [], []
            for i, target_n in enumerate(next(os.walk(self.path))[1]):
                try:
                    target = rdmolfiles.MolFromPDBFile(self.path + target_n + '/receptor.pdb')
                    targetc = target.GetConformer().GetPositions()
                except AttributeError:
                    print("Failed to load target: " + target_n)
                    continue
                for file, value in (("actives", True), ("decoys", False)):
                    with open(self.path + target_n + '/' + file + '_final.ism', 'r') as f:
                        lines = f.readlines()
                        smiless = [x.strip().split(" ")[0] for x in lines]
                        out = Parallel(n_jobs=4)(delayed(Dataset._load_molecule)(smile) for smile in smiless)
                        x_, c_ = zip(*out)
                        x.extend(x_)
                        c.extend(c_)
                        tx.extend([target] * len(x_))
                        tc.extend([targetc] * len(x_))
                        y.extend([value] * len(x_))

                self.target_size = max([t.GetNumAtoms() for t in tx])
                self.molecule_size = max([m.GetNumAtoms() for m in x])

        else:
            for file, value in (("actives", True), ("decoys", False)):
                with open(self.path + file + '_final.ism', 'r') as f:
                    lines = f.readlines()
                    smiless = [x.strip().split(" ")[0] for x in lines]
                    out = Parallel(n_jobs=4)(delayed(Dataset._load_molecule)(smile) for smile in smiless)
                    x, c = zip(*out)
                    y = [value] * len(x)

            self.molecule_size = max([m.GetNumAtoms() for m in x])

        self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y)
        pkl_dict = {'mols': self.mols,
                    'coords': self.coords,
                    'target': self.target}
        pkl.dump(pkl_dict, open(self.path + "dataset.pkl", "wb"))

        # Shuffle data
        idx = np.random.permutation(len(self.mols))
        self.mols, self.coords, self.target = self.mols[idx], self.coords[idx], self.target[idx]

        # Split data
        spl1 = int(len(self.mols) * 0.2)
        spl2 = int(len(self.mols) * 0.1)

        self.x = {"train": self.mols[spl1:],
                  "valid": self.mols[spl2:spl1],
                  "test": self.mols[:spl2]}
        self.c = {"train": self.coords[spl1:],
                  "valid": self.coords[spl2:spl1],
                  "test": self.coords[:spl2]}
        self.y = {"train": self.target[spl1:],
                  "valid": self.target[spl2:spl1],
                  "test": self.target[:spl2]}

        if self.dataset is None:
            self.targets, self.t_coords = np.array(tx), np.array(tc)
            pkl_dict = {'mols': np.array(x),
                        'coords': np.array(c),
                        'target': np.array(y),
                        'targets': self.targets,
                        't_coords': self.t_coords}
            pkl.dump(pkl_dict, open(self.path + "dataset.pkl", "wb"))

            # Shuffle data
            self.targets, self.t_coords = self.targets[idx], self.t_coords[idx]

            # Split data
            spl1 = int(len(self.mols) * 0.2)
            spl2 = int(len(self.mols) * 0.1)

            self.tx = {"train": self.targets[spl1:],
                       "valid": self.targets[spl2:spl1],
                       "test": self.targets[:spl2]}
            self.tc = {"train": self.t_coords[spl1:],
                       "valid": self.t_coords[spl2:spl1],
                       "test": self.t_coords[:spl2]}
    @staticmethod
    def _load_molecule(smiles):
        try:
            # Optimize molecule with MMFF94
            m = Chem.MolFromSmiles(smiles)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            AllChem.MMFFOptimizeMolecule(m)

            return m, m.GetConformer().GetPositions()
        except:
            print("Failed to optimize: " + smiles)

    def reload_dataset(self, path):
        pkl_dict = pkl.load(open(path, 'rb'))
        self.mols = pkl_dict['mols']
        self.coords = pkl_dict['coords']
        self.target = pkl_dict['target']
        self.molecule_size = max([m.GetNumAtoms() for m in self.mols])

        # Shuffle data
        idx = np.random.permutation(len(self.mols))
        self.mols, self.coords, self.target = self.mols[idx], self.coords[idx], self.target[idx]

        # Split data
        spl1 = int(len(self.mols) * 0.2)
        spl2 = int(len(self.mols) * 0.1)

        self.x = {"train": self.mols[spl1:],
                  "valid": self.mols[spl2:spl1],
                  "test": self.mols[:spl2]}
        self.c = {"train": self.coords[spl1:],
                  "valid": self.coords[spl2:spl1],
                  "test": self.coords[:spl2]}
        self.y = {"train": self.target[spl1:],
                  "valid": self.target[spl2:spl1],
                  "test": self.target[:spl2]}

    def save_dataset(self, path, pred=None, target="test", filename=None):
        mols = []
        for idx, (x, c, y) in enumerate(zip(self.x[target], self.c[target], self.y[target])):
            x.SetProp("true", str(y * self.std + self.mean))
            if pred is not None:
                x.SetProp("pred", str(pred[idx][0] * self.std + self.mean))
            mols.append(x)

        if filename is not None:
            w = Chem.SDWriter(path + filename + ".sdf")
        else:
            w = Chem.SDWriter(path + target + ".sdf")
        for mol in mols:
            if mol is not None:
                w.write(mol)

    def replace_dataset(self, path, subset="test", target_name="target"):
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(path)

        for mol in mols:
            if mol is not None:
                # Multitask
                if type(target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in target_name])
                    self.outputs = len(self.target_name)

                # Singletask
                elif target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(target_name))
                    if _y == -1:
                        continue
                    else:
                        y.append(_y)

                else:
                    continue

                x.append(mol)
                c.append(mol.GetConformer().GetPositions())

        # Normalize
        x = np.array(x)
        c = np.array(c)
        y = (np.array(y) - self.mean) / self.std

        self.x[subset] = x
        self.c[subset] = c
        self.y[subset] = y.astype(int) if self.task != "regression" else y

    def set_features(self, use_atom_symbol=True, use_degree=True, use_hybridization=True, use_implicit_valence=True,
                     use_partial_charge=False, use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True,
                     use_acid_base=True, use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, **kwargs):

        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        # Calculate number of features
        mp = MPGenerator([], [], [], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()

    def generator(self, target, task=None):
        return MPGenerator(self.x[target], self.c[target], self.y[target], self.batch,
                           task=task if task is not None else self.task,
                           num_atoms=self.molecule_size,
                           use_atom_symbol=self.use_atom_symbol,
                           use_degree=self.use_degree,
                           use_hybridization=self.use_hybridization,
                           use_implicit_valence=self.use_implicit_valence,
                           use_partial_charge=self.use_partial_charge,
                           use_formal_charge=self.use_formal_charge,
                           use_ring_size=self.use_ring_size,
                           use_hydrogen_bonding=self.use_hydrogen_bonding,
                           use_acid_base=self.use_acid_base,
                           use_aromaticity=self.use_aromaticity,
                           use_chirality=self.use_chirality,
                           use_num_hydrogen=self.use_num_hydrogen)


class MPGenerator(Sequence):
    def __init__(self, x_set, c_set, y_set, batch, task="binary", num_atoms=0,
                 use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                 use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                 use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):
        self.x, self.c, self.y = x_set, c_set, y_set

        self.batch = batch
        self.task = task
        self.num_atoms = num_atoms

        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_c = self.c[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]

        if self.task == "category":
            return self.tensorize(batch_x, batch_c), to_categorical(batch_y)
        elif self.task == "binary":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=int)
        elif self.task == "regression":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=float)
        elif self.task == "input_only":
            return self.tensorize(batch_x, batch_c)

    def tensorize(self, batch_x, batch_c):
        atom_tensor = np.zeros((len(batch_x), self.num_atoms, self.get_num_features()))
        adjm_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms))
        posn_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms, 3))

        for mol_idx, mol in enumerate(batch_x):
            Chem.RemoveHs(mol)
            mol_atoms = mol.GetNumAtoms()

            # Atom features
            atom_tensor[mol_idx, :mol_atoms, :] = self.get_atom_features(mol)

            # Adjacency matrix
            adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")

            # Normalize adjacency matrix by D^(-1/2) * A_hat * D^(-1/2), Kipf et al. 2016
            adjms += np.eye(mol_atoms)
            degree = np.array(adjms.sum(1))
            deg_inv_sqrt = np.power(degree, -0.5)
            deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(deg_inv_sqrt)

            adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)

            adjm_tensor[mol_idx, : mol_atoms, : mol_atoms] = adjms

            # Relative position matrix
            for atom_idx in range(mol_atoms):
                pos_c = batch_c[mol_idx][atom_idx]

                for neighbor_idx in range(mol_atoms):
                    pos_n = batch_c[mol_idx][neighbor_idx]

                    # Direction should be Neighbor -> Center
                    n_to_c = [pos_c[0] - pos_n[0], pos_c[1] - pos_n[1], pos_c[2] - pos_n[2]]
                    posn_tensor[mol_idx, atom_idx, neighbor_idx, :] = n_to_c

        return [atom_tensor, adjm_tensor, posn_tensor]

    def get_num_features(self):
        mol = Chem.MolFromSmiles("CC")
        return len(self.get_atom_features(mol)[0])

    def get_atom_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)

        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())

        ring = mol.GetRingInfo()

        m = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)

            o = []
            o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                            'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']) if self.use_atom_symbol else []
            o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if self.use_degree else []
            o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2]) if self.use_hybridization else []
            o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
            o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
            # o += [atom.GetProp("_GasteigerCharge")] if self.use_partial_charge else [] # some molecules return NaN
            o += [atom.GetIsAromatic()] if self.use_aromaticity else []
            o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                  ring.IsAtomInRingOfSize(atom_idx, 4),
                  ring.IsAtomInRingOfSize(atom_idx, 5),
                  ring.IsAtomInRingOfSize(atom_idx, 6),
                  ring.IsAtomInRingOfSize(atom_idx, 7),
                  ring.IsAtomInRingOfSize(atom_idx, 8)] if self.use_ring_size else []
            o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []

            if self.use_chirality:
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
                except:
                    o += [False, False] + [atom.HasProp("_ChiralityPossible")]
            if self.use_hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
            if self.use_acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]

            m.append(o)

        return np.array(m, dtype=float)
