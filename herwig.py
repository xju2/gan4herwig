import numpy as np

from utils import read_dataframe
from utils import split_to_float
from utils import InputScaler
from utils import boost


def convert_cluster_decay(filename, outname, mode=2,
    with_quark=False, with_pert=False, example=False, do_check_only=False):
    """
    This function reads the original cluster decay files produced by Rivet,
    and it boosts the two hadron decay prodcuts to the cluster frame in which
    they are back-to-back. Save the output to a new file for training the model.
    
    This Herwig dataset is for the "ClusterDecayer" study.
    Each event has q1, q1, cluster, h1, h2.
    I define 3 modes:
    0) both q1, q2 are with Pert=1
    1) only one of q1 and q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    3) at least one quark with Pert=1
    
    Args:
        filename: the original file name
        outname: the output file name
        mode: the mode of the dataset
        with_quark (bool): add two quark angles in the center-of-mass frame as inputs
        with_pert (bool): add the `pertinent` flag of the two quarks as inputs
        example (bool): only show one event as an example
        do_check_only: only check the converted data, do not convert
        
    """
    outname = outname+f"_mode{mode}"+"_with_quark" if with_quark else outname+f"_mode{mode}"
    outname = outname+"_with_pert" if with_pert else outname
    if do_check_only:
        check_converted_data(outname)
        return

    print(f'reading from {filename}')
    df = read_dataframe(filename, ";", 'python')

    q1,q2,c,h1,h2 = [split_to_float(df[idx]) for idx in range(5)]

    if mode == 0:
        selections = (q1[5] == 1) & (q2[5] == 1)
        print("mode 0: both q1, q2 are with Pert=1")
    elif mode == 1:
        selections = ((q1[5] == 1) & (q2[5] == 0)) | ((q1[5] == 0) & (q2[5] == 1))
        print("mode 1: only one of q1 and q2 is with Pert=1")
    elif mode == 2:
        selections = (q1[5] == 0) & (q2[5] == 0)
        print("mode 2: neither q1 nor q2 are with Pert=1")
    elif mode == 3:
        selections = ~(q1[5] == 0) & (q2[5] == 0)
        print("mode 3: at least one quark with Pert=1")
    else: 
        ## no selections
        selections = slice(None)
        print(f"mode {mode} is not known! We will use all events.")

    cluster = c[[1, 2, 3, 4]][selections].values
    
    h1_types = h1[[0]][selections]
    h2_types = h2[[0]][selections]
    h1 = h1[[1, 2, 3, 4]][selections]
    h2 = h2[[1, 2, 3, 4]][selections]


    ## to tell if the quark info is perserved to hadrons
    pert1 = q1[5][selections]
    pert2 = q2[5][selections]

    q1 = q1[[1, 2, 3, 4]][selections]
    q2 = q2[[1, 2, 3, 4]][selections]

    if with_quark:
        org_inputs = np.concatenate([cluster, q1, q2, h1, h2], axis=1)
    else:
        org_inputs = np.concatenate([cluster, h1, h2], axis=1)

    if example:
        print(org_inputs[0])
        return 

    new_inputs = np.array([boost(row) for row in org_inputs])

    def get_angles(four_vector):
        _,px,py,pz = [four_vector[:, idx] for idx in range(4)]
        pT = np.sqrt(px**2 + py**2)
        phi = np.arctan(px/py)
        theta = np.arctan(pT/pz)
        return phi, theta

    out_4vec = new_inputs[:, -4:]
    _,px,py,pz = [out_4vec[:, idx] for idx in range(4)]
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)
    phi, theta = get_angles(new_inputs[:, -4:])
    
    out_truth = np.stack([phi, theta], axis=1)
    cond_info = cluster
    if with_quark:
        print("add quark information")
        ## <NOTE, assuming the two quarks are back-to-back, xju>
        q_phi, q_theta = get_angles(new_inputs[:, 4:8])
        quark_angles = np.stack([q_phi, q_theta], axis=1)
        cond_info = np.concatenate([cond_info, quark_angles], axis=1)

    if with_pert:
        print("add pert information")
        pert_inputs = np.stack([pert1, pert2], axis=1)
        cond_info = np.concatenate([cond_info, pert_inputs], axis=1)

    scaler = InputScaler()
    
    # cond_info: conditional information
    # out_truth: the output hadron angles
    cond_info = scaler.transform(cond_info, outname+"_scalar_input4vec.pkl")
    out_truth = scaler.transform(out_truth, outname+"_scalar_outtruth.pkl")
    
    # add hadron types to the output, [phi, theta, type1, type2]
    out_truth = np.concatenate([out_truth, h1_types, h2_types], axis=1)
    np.savez(outname, cond_info=cond_info, out_truth=out_truth)


def check_converted_data(outname):
    import matplotlib.pyplot as plt

    arrays = np.load(outname+".npz")
    truth_in = arrays['out_truth']
    plt.hist(truth_in[:, 0], bins=100, histtype='step', label='phi')
    plt.hist(truth_in[:, 1], bins=100, histtype='step', label='theta')
    plt.savefig("angles.png")

    scaler_input = InputScaler().load(outname+"_scalar_input4vec.pkl")
    scaler_output = InputScaler().load(outname+"_scalar_outtruth.pkl")

    print("//---- inputs ----")
    scaler_input.dump()
    print("//---- output ----")
    scaler_output.dump()
    print("Total entries:", truth_in.shape[0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert herwig decayer')
    add_arg = parser.add_argument
    add_arg('inname', help='input filename')
    add_arg('outname', help='output filename')
    add_arg('-m', '--mode', help='mode', type=int, default=2)
    add_arg('-q', '--with-quark', help='add quark angles', action='store_true')
    add_arg("-c", '--check', action='store_true', help="check outputs")
    add_arg("-e", '--example', action='store_true', help='print an example event')
    add_arg("-p", '--with-pert', action='store_true', help='add perturbation information')
    args = parser.parse_args()
    

    convert_cluster_decay(args.inname, args.outname,
        args.mode, args.with_quark, args.with_pert, args.example, args.check)