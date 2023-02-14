from models.bert import BertClf
from models.image import ImageClf
from models.multi_model import MultimodalBertClf






from models.mmbt_drop import DROPMultimodalBertClf


MODELS = {
    "multi_model": MultimodalBertClf
}


def get_model(args):
    return MODELS[args.model](args)
