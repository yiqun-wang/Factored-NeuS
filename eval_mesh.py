import argparse
from evaluation import dtu_eval
from pathlib import Path
from pyhocon import ConfigFactory


parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, required=True)  # 65
parser.add_argument('--setting', type=str, required=True)  # womask
parser.add_argument("--suffix", default="")  # 00300000
# parser.add_argument("--confname")

args = parser.parse_args()
scene = args.scene
setting = args.setting
# expname = Path(args.confname).with_suffix("").name

evaldir = Path(f"exp/data_DTU/dtu_scan{args.scene}/{setting}/meshes_clean")
inp_mesh_path = evaldir / '{:0>8d}.ply'.format(int(args.suffix)) # f"{args.suffix}.ply"

dtu_eval.eval(inp_mesh_path, int(scene), "public_data/dtu_eval", evaldir, args.suffix)

# conf = ConfigFactory.parse_file(args.confname)
# if conf["dataset"]["data_dir"] == "DTU":
#     dtu_eval.eval(inp_mesh_path, int(scene), "data/dtu_eval", evaldir, args.suffix)
# else:
#     epfl_eval.eval(inp_mesh_path, scene, "data/epfl", evaldir, args.suffix)