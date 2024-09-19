import torch

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

def safe_log1p_exp(x, limit=80):
    # more stable version of log(1 + exp(x)), avoid inf values
    return torch.where(x < limit, torch.log1p(torch.exp(x)), x)

def getattr_rec(obj, name):
    path = name.split(".")
    current_obj = obj

    for prop in path:
        current_obj = getattr(current_obj, prop)

    return current_obj

def setattr_rec(obj, name, value):
    path = name.split(".")
    current_obj = obj
    for index, prop in enumerate(path):
        print(current_obj)
        if type(current_obj) == dict:
            if index != len(path)-1:
                current_obj = current_obj[prop]
        else:
            current_obj = getattr(current_obj, prop)
    if type(current_obj == dict):
        current_obj[path[-1]] = value
    else:
        current_obj = value


def get_state_dict_from_TF_checkpoint(pretrain_path, logger):

    from tensorflow import compat
    from tensorflow.python.framework.errors_impl import NotFoundError
    TF = compat.v1

    # return value
    state_dict = {} 

    # read TF checkpoint
    TF.disable_eager_execution()
    reader = TF.train.load_checkpoint(pretrain_path)

    # helper functions
    # -------------------------------------------------------------------------------------------------------------
    #
    def get_torch_value(tf_path):
        """
        gets the value from the specified path and handles the conversion to PT format
        """
        tf_value = reader.get_tensor(tf_path)

        if len(tf_value.shape) == 4:
            torch_value = torch.from_numpy(tf_value.transpose((3, 2, 0, 1)))
        elif len(tf_value.shape) == 1:
            torch_value = torch.from_numpy(tf_value)
        else:
            raise Exception("Unknown tesnor shape in get_state_dict_from_TF_checkpoint.get_torch_value!")
        
        return torch_value
    #
    def init_pt_batch_norm(pt_prefix, tf_prefix):
        """
        Initializes the BatchNorm at the given location with all BN parameters.

        !!! No trailing slashes/dots! They are added in here.
        """
        state_dict[pt_prefix + ".weight"]               = get_torch_value(tf_prefix + "/BatchNorm/gamma")
        state_dict[pt_prefix + ".bias"]                 = get_torch_value(tf_prefix + "/BatchNorm/beta")
        state_dict[pt_prefix + ".running_mean"]         = get_torch_value(tf_prefix + "/BatchNorm/moving_mean")
        state_dict[pt_prefix + ".running_var"]          = get_torch_value(tf_prefix + "/BatchNorm/moving_variance")
        state_dict[pt_prefix + ".num_batches_tracked"]  = torch.zeros(1) # this should result in the same behaviour as TF
    # -------------------------------------------------------------------------------------------------------------

    # Backbone
    # -------------------------------------------------------------------------------------------------------------
    #
    # PreBlocks
    state_dict["backbone.conv1.weight"]             = get_torch_value("resnet_v1_50/conv1/weights")
    init_pt_batch_norm("backbone.bn1", "resnet_v1_50/conv1")
    #
    # Blocks
    bottleneck_quantity = [None, 3, 4, 6, 3] # start counting at 1
    #
    for resnet_block_index in range(1, 5): # layerX / blockX
        for bottleneck_index in range(bottleneck_quantity[resnet_block_index]): # layerX.Y / blockX/unit_[Y+1]
    #        
            pt_prefix = "backbone.layer" + str(resnet_block_index) + "." + str(bottleneck_index)
            tf_prefix = "resnet_v1_50/block" + str(resnet_block_index) + "/unit_" + str(bottleneck_index + 1) + "/bottleneck_v1"
    #
            # downsample / shortcut is only present in first block of layer / unit of block (PT/TF)
            if bottleneck_index == 0:
                # convolution
                state_dict[pt_prefix + ".downsample.0.weight"] = get_torch_value(tf_prefix + "/shortcut/weights")
                # BatchNorm
                init_pt_batch_norm( pt_prefix + ".downsample.1", tf_prefix + "/shortcut")
    #       
            for i in range(1, 4):
                # conv_i
                state_dict[pt_prefix + ".conv" + str(i) + ".weight"] = get_torch_value(tf_prefix + "/conv" + str(i) + "/weights")
                # bn_i
                init_pt_batch_norm(pt_prefix + ".bn" + str(i), tf_prefix + "/conv" + str(i))
    # -------------------------------------------------------------------------------------------------------------

    # Head
    # -------------------------------------------------------------------------------------------------------------
    #
    # variance estimator
    try:
        state_dict["heads.sig_layer.weight"] = get_torch_value("resnet_v1_50/Distributions/sig/weights")
        state_dict["heads.sig_layer.bias"]   = get_torch_value("resnet_v1_50/Distributions/sig/biases")
    except NotFoundError:
        logger.info("Sig layer not found. This is expected if training from pretrain but not expected when trying to eval.")
    #
    # logits
    state_dict["heads.logit_layer.weight"] = get_torch_value("resnet_v1_50/logits/weights")
    state_dict["heads.logit_layer.bias"]   = get_torch_value("resnet_v1_50/logits/biases")
    # -------------------------------------------------------------------------------------------------------------

    return state_dict