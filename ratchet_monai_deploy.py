import os
import shutil
import tempfile
import glob
import PIL.Image
import torch
import numpy as np

from ignite.engine import Events

from monai.apps import download_and_extract
from monai.config import print_config
from monai.networks.nets import DenseNet121
from monai.engines import SupervisedTrainer
from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)
from monai.utils import set_determinism

set_determinism(seed=0)

print_config()

import monai.deploy.core as md  # 'md' stands for MONAI Deploy (or can use 'core' instead)
from monai.deploy.core import (
    Application,
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)

from datasets.mimic import CustomImageDataset
from model.transformer import Transformer
from model.utils import create_target_masks

from tokenizers import ByteLevelBPETokenizer

'''
STEP 1: LOAD DATASET
'''
csv_root = 'preprocessing/mimic'
img_dir = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org'

# Parameters
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}
max_epochs = 100

tokenizer = ByteLevelBPETokenizer(
    os.path.join(csv_root, 'mimic-vocab.json'),
    os.path.join(csv_root, 'mimic-merges.txt'),
)


@md.input("image", DataPath, IOType.DISK)
@md.output("image", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["pillow"])
class LoadPILOperator(Operator):
    """Load image from the given input (DataPath) and set numpy array to the output (Image)."""

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        import numpy as np
        from PIL import Image as PILImage

        input_path = op_input.get().path
        if input_path.is_dir():
            input_path = next(input_path.glob("*.*"))  # take the first file

        image = PILImage.open(input_path)
        image = image.convert("L")  # convert to greyscale image
        image = image.resize((224, 224))
        image_arr = np.asarray(image)

        output_image = Image(image_arr)  # create Image domain object with a numpy array
        op_output.set(output_image)



@md.input("image", Image, IOType.IN_MEMORY)
@md.output("output", DataPath, IOType.DISK)
@md.env(pip_packages=["monai"])
class TransformerOperator(Operator):
    """Classifies the given image and returns the class name."""

    @property
    def transform(self):
        return Compose([AddChannel(), ScaleIntensity(), EnsureType()])

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext, max_length=128):
        import json

        import torch

        img = op_input.get().asnumpy()
        img = np.divide(img, 255.).astype('float32')
        #img = img[None, ...]

        image_tensor = self.transform(img)  
        image_tensor = image_tensor[None].float()  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        output = torch.from_numpy(np.array([[tokenizer.token_to_id('<s>')]], dtype=np.int32)).to(device)
        inp_img = torch.cat(3 * [image_tensor], dim=1).to(device)

        transformer = context.models.get()

        for i in range(max_length):
        	combined_mask = create_target_masks(output).to(device)

        	predictions, attention_weights = transformer(inp_img,
                                                     output,
                                                     False,
                                                     combined_mask,
                                                     None)

       		# select the last word from the seq_len dimension
        	predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        	predicted_id = torch.argmax(predictions, dim=-1)

        	# return the result if the predicted_id is equal to the end token
        	if predicted_id == 2:
           		break

        	# concatentate the predicted_id to the output which is given to the decoder
        	# as its input.
        	output = torch.cat([output, predicted_id], dim=-1)

        out = torch.squeeze(output, dim=0)[1:]#, attention_weights
        print(out)

        '''

        model = context.models.get()  # get a TorchScriptModel object

        with torch.no_grad():
            outputs = model(image_tensor)

      
        print(result)

        # Get output (folder) path and create the folder if not exists
        output_folder = op_output.get().path
        output_folder.mkdir(parents=True, exist_ok=True)

        # Write result to "output.json"
        output_path = output_folder / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)
        '''


@md.resource(cpu=1, gpu=1, memory="1Gi")
class App(Application):
    """Application class for the MedNIST classifier."""

    def compose(self):
        load_pil_op = LoadPILOperator()
        transformer_op = TransformerOperator()

        self.add_flow(load_pil_op, transformer_op)


if __name__ == "__main__":
    App(do_run=True)


