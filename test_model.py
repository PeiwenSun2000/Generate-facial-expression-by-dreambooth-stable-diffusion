from diffusers import StableDiffusionPipeline
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm
import os
def sample(prompt_list,seed,num_per_prompt=10):
    torch.manual_seed(seed)

    pipe = StableDiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            scheduler=DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=True,
            ),
            safety_checker=None
        )

    pipe = pipe.to("cuda")
    output_dir="/mnt/tempDisk/samples"
    for prompt in tqdm(prompt_list):
        images = pipe(prompt, width=256, height=256, num_inference_steps=30, num_images_per_prompt=num_per_prompt).images
        image_dir=os.path.join(output_dir,prompt+"-"+str(seed))
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i, image in enumerate(images):
            image.save(os.path.join(image_dir,"test-"+str(i)+".png"))

model_path = "/mnt/tempDisk/model/au_nlp_model-Step-8000"  
prompt_list = ["Asian,female,close-up,whole-face,front-face,young,upper-lip-raiser,lip-corner-puller",
    "Africa,male,close-up,whole-face,front-face,young,upper-lip-raiser,lip-corner-puller",
    "European,female,close-up,whole-face,front-face,young,inner-brow-raiser,cheek-raiser",
    "Asian,female,close-up,whole-face,front-face,young,lip-corner-depressor",
    "European,female,close-up,whole-face,front-face,young,lip-corner-depressor,lip-pressor"]
sample(prompt_list,111,30)
sample(prompt_list,222,30)
sample(prompt_list,333,30)


