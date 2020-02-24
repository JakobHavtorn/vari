import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=1, input_dtype=torch.FloatTensor, device=None):

    def register_hook(module):
        # TODO Make this hooking aware of the depth within the model
        #      Is this a top module or a module within a module etc. Display the depth in some way ()

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            elif isinstance(output, torch.distributions.Distribution):
                summary[m_key]["output_shape"] = [batch_size]
                summary[m_key]["output_shape"] += list(output.event_shape)
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    if device is None:
        device = next(model.parameters()).device  # Device of first parameters in model
    x = [torch.rand(batch_size, *in_size).to(device) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # TODO: Return and print this as a pandas dataframe
    print("----------------------------------------------------------------------------------------------")
    line_new = "{:>25}  {:>25} {:>25} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
    print(line_new)
    print("==============================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>25}  {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = np.prod(input_size) * batch_size * 4. / 1e6
    total_output_size = total_output * 4. / 1e6  # x2 for gradients
    total_params_size = total_params.item() * 4. / 1e6
    total_gradients_size = total_params.item() * 4. / 1e6
    total_size = total_params_size + total_gradients_size + total_output_size + total_input_size

    print("==============================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward pass size (MB): %0.2f" % total_output_size)
    print("Parameters size (MB): %0.2f" % total_params_size)
    print("Gradients size (MB): %0.2f" % total_gradients_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------------------------------------")
    # return summary
