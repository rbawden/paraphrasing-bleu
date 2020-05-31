# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


# Get the device indicated by the first device_id
def get_device(device_ids):
    if len(device_ids) == 0:
        return torch.device("cpu")
    return torch.device("cuda:{}".format(device_ids[0]))


# Put inputs into devices
def putter(inputs, device_ids):
    assert len(inputs) <= len(device_ids), \
        "input should be less or equal to devices"

    outputs = []
    for device_id, feature in zip(device_ids, inputs):
        device = torch.device("cuda:{}".format(device_id))

        feature = map_structure(
            lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
            feature
        )

        outputs.append(feature)

    return outputs


# Simulate the tensorflow's map_structure
def map_structure(func, states):
    def _map_structure(obj):
        if isinstance(obj, list):
            return [_map_structure(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple([_map_structure(v) for v in obj])
        elif isinstance(obj, dict):
            return dict([(k, _map_structure(v)) for k, v in obj.items()])
        else:
            return func(obj)

    return _map_structure(states)


# model parallelization
def parallel_model(model, features, devices):
    if len(devices) == 1:
        device = get_device(devices)
        model = model.to(device)
        feature = map_structure(
            lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
            features[0]
        )
        output = model(feature)

        return putter([output], devices)

    models = nn.parallel.replicate(model, devices)
    features = putter(features, devices)
    outputs = nn.parallel.parallel_apply(
        models[:len(features)], features)

    return putter(outputs, [devices[0]] * len(outputs))
