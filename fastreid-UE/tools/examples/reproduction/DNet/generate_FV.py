""" --------------------------------------------------------------------------- BA_UE
based on train_net.py, this file is used to generate a feature vector for a specific image
"""

#!/usr/bin/env python
# encoding: utf-8
"""
@author:  Andreas Gebhardt
@contact: AGebhardt1999@gmail.com
"""

import json
import os
import sys
import numpy as np
import torch
import h5py

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)


    if args.eval_only:
        cfg.defrost()
        model = DefaultTrainer.build_model(cfg)

        data_loader, evaluator = DefaultTrainer.build_evaluator(cfg, eval(str(cfg.DATASETS.TESTS))[0])

        test_image = None
        break_outer_loop = False

        for input in data_loader:
            #print(input["images"][0].shape)
            for index, path in enumerate(input["img_paths"]):
                if path.split("/")[-1] == "0001_c1s1_001051_03.jpg":
                    print("found", path, "at index", index)
                    #import torch
                    #import numpy as np
                    #from PIL import Image
                    #image = Image.fromarray((255 * input["images"][index].permute(1, 2, 0).cpu().numpy()).astype(np.uint8))
                    #image.save("confirm_image.png")
                    #test_image = input["images"]
                    test_image = input["images"][index]
                    break_outer_loop = True
                    break
            if break_outer_loop:
                break
        
        test_image = test_image.unsqueeze(0)
        print(test_image.shape)
        model.eval()

        def load_hdf5(fname):
            hf = h5py.File(fname, 'r')
            d = {key: np.array(hf.get(key)) for key in hf.keys()}
            hf.close()
            return d

        print("Loading h5 file...")
        h5_dict = load_hdf5("/datasets_nas/ange8547/Market/bounding_box_test.h5")
        print(h5_dict.keys())

        print(torch.from_numpy(h5_dict["image_data"][0].transpose(2, 0, 1)).unsqueeze(0).shape)
        test_image_tf = torch.from_numpy(h5_dict["image_data"][0].transpose(2, 0, 1), ).float().unsqueeze(0).to("cuda:0")
        print(test_image_tf[0][0][0][0].__class__.__name__)
        print(h5_dict["image_data"][0][0][0][0].__class__.__name__)

        
        # still has differences with correct interpolation
        print("test_image:", test_image)
        print("test_image_tf:", test_image_tf)
        diff = test_image - test_image_tf
        print("diff: ", diff)

        diff = diff.abs()
        # Count the number of non-zero entries
        non_zero_count = torch.nonzero(diff).size(0)

        # Measure the maximum, average, and sum of values of the non-zero entries
        non_zero_values = diff[diff != 0]
        max_non_zero = non_zero_values.max().item()
        avg_non_zero = non_zero_values.mean().item()
        sum_non_zero = non_zero_values.sum().item()

        print("Number of non-zero entries:", non_zero_count)
        print("Maximum value of non-zero entries:", max_non_zero)
        print("Average value of non-zero entries:", avg_non_zero)
        print("Sum of values of non-zero entries:", sum_non_zero)

        # Count the number of non-zero pixels in each channel
        non_zero_counts = (diff != 0).sum(dim=(2, 3))  # Sum along height and width dimensions

        # Calculate the average number of non-zero pixels across channels
        avg_non_zero_count = non_zero_counts.float().mean(dim=1).item()  # Mean along the channel dimension

        print("Number of non-zero pixels in each channel:", non_zero_counts)
        print("Average number of non-zero pixels across channels:", avg_non_zero_count)


        from PIL import Image
        image = Image.fromarray(((diff).squeeze().permute(1, 2, 0).cpu().numpy() *40).astype(np.uint8))
        image.save("confirm_image.png")
        

        mu_out, var_out = model.forward(test_image)
        mu_out_tf, var_out_tf = model.forward(test_image_tf)

        print(mu_out)
        print(mu_out_tf)

        tf_FV_path = os.path.join(cfg.MODEL.BACKBONE.PRETRAIN_PATH[1:], "example_FV.json")
        with open(tf_FV_path, 'r') as f:
            data = json.load(f)

        #print(data["mean"])
        #np.set_printoptions(threshold=np.inf)
        print((data["mean"] - mu_out_tf.cpu().detach().numpy()))
        print((data["var"] - var_out_tf.cpu().detach().numpy()))

        mean_diff = data["mean"] - mu_out.cpu().detach().numpy()
        var_diff = data["var"] - var_out.cpu().detach().numpy()
        mean_diff_tf = data["mean"] - mu_out_tf.cpu().detach().numpy()
        var_diff_tf = data["var"] - var_out_tf.cpu().detach().numpy()

        mean_diff_diff = mean_diff - mean_diff_tf
        var_diff_diff = var_diff - var_diff_tf

        for vec, desc in zip([mean_diff, var_diff, mean_diff_tf, var_diff_tf, mean_diff_diff, var_diff_diff], 
                ["difference in mean between TF model output and PT model output using normal Market",
                "difference in var between TF model output and PT model output using normal Market",
                "difference in mean between TF model output and PT model output using DNet-Market",
                "difference in var between TF model output and PT model output using DNet-Market",
                "difference between the two mean-diff vectors",
                "difference between the two var-diff vectors"]):
            vec = np.abs(vec)
            max_val = np.max(vec)
            avg_val = np.mean(vec)
            l2_norm = np.linalg.norm(vec)
            print("----", desc)
            print("Maximum value:", max_val)
            print("Average value:", avg_val)
            print("L2 norm:", l2_norm)


        exit()

        print(output["features"].shape, output["features"].__class__.__name__)
        print(output["features_sig"].shape, output["features_sig"].__class__.__name__)

        mu_out = np.flip(output["features"].cpu().detach().numpy(), axis=0)
        var_out = np.flip(output["features_sig"].cpu().detach().numpy(), axis=0)

        mu_tf = np.array(data["mean"])
        var_tf = np.array(data["var"])

        print("mu equal:", mu_out - mu_tf)
        print("var equal:", var_out - var_tf)

        grad_fn = output["features"].grad_fn
        while grad_fn is not None:
            print(type(grad_fn).__name__)
            for input_tensor in grad_fn.next_functions:
                if input_tensor[0] is not None:
                    if isinstance(grad_fn, torch.autograd.function.SubBackward) or isinstance(grad_fn, torch.autograd.function.MeanBackward):
                        print("\tInput value: ", input_tensor[0])
                    else:
                        print("\tInput value: ", input_tensor[0].variable)
            grad_fn = grad_fn.next_functions[0][0].grad_fn
        
        exit()

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


#python3 tools/DNet/generate_FV.py --eval-only --config-file ./configs/DNet.yml MODEL.DEVICE "cuda:0"