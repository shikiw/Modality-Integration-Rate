import argparse
import torch
import os
import json
import math
from tqdm import tqdm
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from mir_util import *




parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--base_llm", type=str, default=None)
parser.add_argument("--text_data_path", type=str, default="")
parser.add_argument("--image_data_path", type=str, default="")
parser.add_argument("--eval_num", type=int, default=100)
parser.add_argument("--mode", type=str, default="fast")
args = parser.parse_args()


### Model ###
disable_torch_init()
model_path = args.model_path
model_path = os.path.expanduser(model_path)
model_base = args.base_llm
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)


def read_story_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    parts = content.split('@highlight')
    story = parts[0].strip()
    highlights = [part.strip() for part in parts[1:]]
    return story, highlights


### Data ###
text_data_path = args.text_data_path
data_texts = os.listdir(text_data_path)

image_base_path = args.image_data_path
data_images = os.listdir(image_base_path)

# NOTE: You can specify your own data for evaluation
# NOTE: For example, we can use images from TextVQA val and text from CNN/DM as follows.
# # cnn/daily mail and textvqa
# text_data_path = "/mnt/hwfile/mllm/huangqidong/nlp/cnn/stories"
# data_texts = os.listdir(text_data_path)
# # TextVQA
# image_base_path = "/mnt/hwfile/mllm/chenlin/llava/data/eval/textvqa/train_images/"
# question_file = "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
# questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
# data_images = [questions[i]["image"] for i in range(len(questions))]


### Get vision/text tokens ###
all_hidden_states = {"vision": [], "text": []}

for idx in tqdm(range(args.eval_num)):
    data_image = data_images[idx]
    data_text = data_texts[idx]

    raw_image = os.path.join(image_base_path, data_image)
    raw_image = Image.open(raw_image)
    raw_image = raw_image.convert("RGB")
    image_tensor = process_images([raw_image], image_processor, model.config)[0]

    # If we use text from CNN/DM, we can process with read_story_file
    # caption = data_image["conversations"][1]["value"]
    caption = read_story_file(os.path.join(text_data_path, data_text))[0]

    qs = ""
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # conv_mode = "llava_v1"
    conv_mode = "vicuna_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], caption)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    # input_ids = torch.tensor(tokenizer(prompt).input_ids)

    # Inference
    image_tensor = image_tensor.unsqueeze(0)
    input_ids = input_ids.to(device='cuda', non_blocking=True).unsqueeze(0)
    image_start_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=raw_image.size,
            do_sample=False,
            num_beams=1,
            max_new_tokens=1,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
        )

    hidden_states = outputs.hidden_states
    latent_hidden_states = [hidden_state.squeeze() for hidden_state in hidden_states[0]]
    # inputs_embeds = hidden_states[0][0]
    # output_ids = outputs[0]
    # output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # You may need to specify the number of image tokens, e.g, 576 for llava-v1.5 7B model
    vision_hidden_states = [latent[image_start_idx:image_start_idx+576,:].detach().cpu() for latent in latent_hidden_states]
    text_hidden_states = [latent[image_start_idx+576:,:].detach().cpu() for latent in latent_hidden_states]
    all_hidden_states["vision"].append(vision_hidden_states) # 100 * [33, 576, 4096]
    all_hidden_states["text"].append(text_hidden_states)


### MIR Eval ###
layer_length = len(all_hidden_states["vision"][0])
plot_data = {"Per-Layer-MIR":[]}
for layer_idx in range(1, layer_length):

    vision_features = [hidden_states[layer_idx].float().cuda() for hidden_states in all_hidden_states["vision"]]
    text_features = [hidden_states[layer_idx].float().cuda() for hidden_states in all_hidden_states["text"]]
    vision_features = torch.cat(vision_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    # Text-Centric Normalization
    scale_factor = 1. / text_features.norm(p=2, dim=-1).mean(0)
    vision_features = scale_factor * vision_features
    text_features = scale_factor * text_features
    # print(f"Scale factor: {scale_factor}")

    # 3-Sigma Outlier Removal
    vision_features = replace_outliers_with_median_l2(vision_features)
    text_features = replace_outliers_with_median_l2(text_features)

    # Switch between fast mode and accurate mode, we use fast mode by default
    if args.mode == "fast":
        plot_data["Per-Layer-MIR"].append(calculate_fid_pytorch(vision_features, text_features))
    else:
        plot_data["Per-Layer-MIR"].append(calculate_fid(vision_features, text_features))

    print("Layer #{}\tPer-Layer MIR: {}".format(layer_idx, plot_data["Per-Layer-MIR"][-1]))

final_mir = math.log(sum(plot_data["Per-Layer-MIR"]), 10)
print(f"Overall MIR: {final_mir}")


