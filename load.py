import os
import torch

# outputs/demo/tennis/hmr4d_results.pt
data_path = os.path.join(os.path.dirname(__file__), "outputs", "demo", "tennis", "hmr4d_results.pt")

data: dict = torch.load(data_path)

# dict_keys(['smpl_params_global', 'smpl_params_incam', 'K_fullimg', 'net_outputs'])
# print(data.keys())

for k, v in data.items():
    print(f"{k}")

    if isinstance(v, torch.Tensor):
        print(f"  type: {type(v)}, shape: {v.shape}, dtype: {v.dtype}")
    elif isinstance(v, dict):
        for sub_k, sub_v in v.items():
            # print the type and shape of the sub-value
            print(
                f"  {sub_k}: type: {type(sub_v)}, shape: "
                f"{sub_v.shape if isinstance(sub_v, torch.Tensor) else type(sub_v)}"
            )
    else:
        print(f"  type: {type(v)}")
