import jax
import jax.numpy as jnp
from einops import rearrange
from transformers import AutoProcessor, FlaxCLIPModel

class CLIPOriginal():
    def __init__(self, clip_model="clip-vit-base-patch32"):
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        self.clip_model = FlaxCLIPModel.from_pretrained(f"openai/{clip_model}")

        self.img_mean = jnp.array(self.processor.image_processor.image_mean)
        self.img_std = jnp.array(self.processor.image_processor.image_std)

    def embed_img(self, img):
        """
        img shape (H W C) and values in [0, 1].
        returns shape (D)
        """
        H, W, C = img.shape
        if H!=224 or W!=224:
            img = jax.image.resize(img, (224, 224, C), method='bilinear')
        img = rearrange((img-self.img_mean)/self.img_std, "H W C -> 1 C H W")
        z_img = self.clip_model.get_image_features(img)[0]
        return z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True)

    def embed_txt(self, prompts):
        """
        prompts is list of strings
        returns shape (B D)
        """
        inputs = self.processor(text=prompts, return_tensors="jax", padding=True)
        z_text = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return z_text / jnp.linalg.norm(z_text, axis=-1, keepdims=True)
    

from modeling_flax_clip import FlaxCLIPVisionModule
class CLIP():
    def __init__(self, clip_model="clip-vit-base-patch32"):
        assert clip_model=="clip-vit-base-patch32", "Only clip-vit-base-patch32 is supported for backpropagation"
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        self.clip_model = FlaxCLIPModel.from_pretrained(f"openai/{clip_model}")

        self.img_mean = jnp.array(self.processor.image_processor.image_mean)
        self.img_std = jnp.array(self.processor.image_processor.image_std)

        self.my_flax_vision = FlaxCLIPVisionModule(self.clip_model.config.vision_config)
        self.my_params = jax.tree.map(lambda x: x.copy(), self.clip_model.params)

        kernel = self.my_params['vision_model']['embeddings']['patch_embedding']['kernel']
        kernel = rearrange(kernel, "H W Di Do -> (H W Di) Do")
        self.my_params['vision_model']['embeddings']['patch_embedding']['kernel'] = kernel

    def embed_txt(self, prompts):
        """
        prompts is list of strings
        returns shape (B D)
        """
        inputs = self.processor(text=prompts, return_tensors="jax", padding=True)
        z_text = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return z_text / jnp.linalg.norm(z_text, axis=-1, keepdims=True)
    
    def embed_img(self, img):
        """
        img shape (H W C) and values in [0, 1].
        returns shape (D)
        """
        H, W, C = img.shape
        if H!=224 or W!=224:
            img = jax.image.resize(img, (224, 224, C), method='bilinear')
        img = rearrange((img-self.img_mean)/self.img_std, "H W C -> 1 H W C")
        # z_img = self.clip_model.get_image_features(img)[0]
        vision_outputs = self.my_flax_vision.apply(dict(params=self.my_params), img)
        pooled_output = vision_outputs[1] # pooled_output
        visual_projection = self.clip_model.params['visual_projection']['kernel']
        z_img = (pooled_output @ visual_projection)[0]
        return z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True)
    