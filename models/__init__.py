#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from models.bert import BertClf
from models.bow import GloveBowClf
from models.concat_bert import MultimodalConcatBertClf
from models.concat_bow import  MultimodalConcatBowClf
from models.image import ImageClf
from models.mmbt import MultimodalBertClf

from models.mmbt_cap import MY_MultimodalBertClf

from models.mmbt_cap_sigmoid import MY_MultimodalBertClf_sigmoid





from models.mmbt_drop import DROPMultimodalBertClf


MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    "mmbt_cap": MY_MultimodalBertClf,
    "mmbt_cap_sigmoid": MY_MultimodalBertClf_sigmoid,
    "mmbt_drop": DROPMultimodalBertClf
}


def get_model(args):
    return MODELS[args.model](args)
