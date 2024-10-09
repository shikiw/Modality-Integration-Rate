import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        prompt = "USER:" + qs + "ASSISTANT:"

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return prompt, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    # input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    in_context_eval = True
    use_multi_image = False
    context_num = 2
    # SEED
    # incontext_data = [
    #     {
    #         "image": "/mnt/hwfile/mllm/chenlin/llava/data/eval/SEED-Bench/SEED-Bench-image/1454426_2591111986",
    #         "question": "How many towels are in the image?\nA. One\nB. Two\nC. Three\nD. Four\nAnswer with the option's letter from the given choices directly.",
    #         "answer": "A",
    #     },
    #     {
    #         "image": "/mnt/hwfile/mllm/chenlin/llava/data/eval/SEED-Bench/SEED-Bench-image/1307737_3736205576",
    #         "question": "What type of building is in the image?\nA. A hotel\nB. A house\nC. A cabin\nD. A shed\nAnswer with the option's letter from the given choices directly.",
    #         "answer": "C",
    #     }
    # ]
    # MME
    incontext_data = [
        {
            "image": "/mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release/code_reasoning/0001.png",
            "question": "The image shows a python code. Is the output of the code 'Hello'?\nAnswer the question using a single word or phrase.",
            "answer": "Yes",
        },
        {
            "image": "/mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release/artwork/images/29266.jpg",
            "question": "Is this artwork created by maris, jacobus hendricus?\nAnswer the question using a single word or phrase.",
            "answer": "No",
        }
    ]

    if in_context_eval:
        # context_qs = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        context_qs = ""
        context_images = []
        for i in range(context_num):
            qas = incontext_data[i]

            image_name = qas["image"]
            instruction = qas["question"]
            answer = qas["answer"]

            if use_multi_image:
                raw_image = Image.open(image_name)
                raw_image = raw_image.convert("RGB")
                image_tensor = process_images([raw_image], image_processor, model.config)[0]
                image_tensor = image_tensor.unsqueeze(0)

                context_images.append(image_tensor.to(dtype=torch.float16, device='cuda:0', non_blocking=True))

            if "\n<image>" in instruction:
                qs_splits = instruction.split("\n<image>")
            else:
                qs_splits = instruction.split("<image>\n")
            qs1 = ""
            for q in qs_splits:
                qs1 += q

            if use_multi_image:
                if model.config.mm_use_im_start_end:
                    qs1 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs1
                else:
                    qs1 = DEFAULT_IMAGE_TOKEN + '\n' + qs1

            prompt = "USER:" + qs1 + "ASSISTANT:" + answer
            context_qs += prompt

    # context_ids = tokenizer_image_token(context_qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

    for (prompt, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        # input_ids = torch.cat([context_ids, input_ids[:, 1:]], dim=-1)
        prompt = context_qs + prompt[0]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.to(device='cuda', non_blocking=True).unsqueeze(0)
        # image_start_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]

        if use_multi_image:
            context_images.append(image_tensor.to(dtype=torch.float16, device='cuda:0', non_blocking=True))
        else:
            context_images = image_tensor.to(dtype=torch.float16, device='cuda:0', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=context_images,
                # image_start_idx=image_start_idx,
                # evict_or_reserve_mode="reserve_topk",
                # evict_or_reserve_token_num=100,
                # evict_or_reserve_threshold=10.0,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
