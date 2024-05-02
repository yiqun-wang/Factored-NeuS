# Factored-NeuS

## Usage

**Environment**

```shell
pip install - r requirements.txt
```

**Training and Evaluation**

```shell
bash sh_dtu.sh
```

You can modify the parameters in Python execution command in the following format.
```shell
python exp_runner.py --mode train --conf ./confs/wmask.conf --case data_DTU/dtu_scan97 --type dtu --gpu 0 --surface_weight 0.1
```
