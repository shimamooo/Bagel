# code_image_dataset.py -- dataset for (code, rendered image) mid-training
# Supports three sequence formats from the data collection pipeline:
#   full        : <prompt><code><vit_image><vae_image>
#   prompt_free : <code><vit_image><vae_image> or <vit_image><vae_image><code>
#   trace       : <code_chunk_1>...<code_chunk_n><vit_image><vae_image>
#                 (proper intermediate renders require pre-rendering; see NOTE below)
import json
import os
import random
import traceback
from PIL import Image, ImageFile

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# NOTE: trace sequences in our JSONL only carry the final rendered image.
# Intermediate per-step images require a separate offline pre-rendering pass
# (render each code prefix and store the image paths in the JSONL).
# Until that is done, trace samples are trained with the final image only.

_TEXT_PLAN = lambda loss, is_code, enable_cfg=1: {
    'type': 'text', 'enable_cfg': enable_cfg, 'loss': loss,
    'is_code': is_code, 'special_token_loss': 0, 'special_token_label': None,
}
_VIT_PLAN = lambda enable_cfg=1: {
    'type': 'vit_image', 'enable_cfg': enable_cfg, 'loss': 0,
    'special_token_loss': 0, 'special_token_label': None,
}
_VAE_PLAN = lambda loss, enable_cfg=1: {
    'type': 'vae_image', 'enable_cfg': enable_cfg, 'loss': loss,
    'special_token_loss': 0, 'special_token_label': None,
}


class CodeImageIterableDataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,           # VAE / generation transform (high-res, stride 16)
        vit_transform,       # ViT / understanding transform (lower-res, stride 14)
        tokenizer,
        jsonl_path_list,
        data_dir_list,       # unused (image paths are absolute in our JSONL)
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        direction="mixed",   # "code2image" | "image2code" | "mixed"
        max_trace_steps=8,
        num_used_data=None,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.direction = direction
        self.max_trace_steps = max_trace_steps
        self.data_status = data_status

        samples = []
        for jsonl_path in jsonl_path_list:
            if not os.path.exists(jsonl_path):
                if local_rank == 0:
                    print(f"WARNING: {jsonl_path} not found, skipping")
                continue
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(line)

        if num_used_data:
            limit = num_used_data[0] if isinstance(num_used_data, list) else num_used_data
            samples = samples[:limit]

        # store as (json_str, row_idx) so set_epoch can shuffle
        self.data_paths = [(s, i) for i, s in enumerate(samples)]
        self.set_epoch()

        if local_rank == 0:
            print(f"{dataset_name}: loaded {len(self.data_paths)} samples from {jsonl_path_list}")

    def _load_image(self, path):
        return pil_img2rgb(Image.open(path))

    def _token_count(self, ids, vit_tensor, vae_tensor):
        vit_h, vit_w = vit_tensor.shape[1], vit_tensor.shape[2]
        vae_h, vae_w = vae_tensor.shape[1], vae_tensor.shape[2]
        vit_tokens = vit_h * vit_w // (self.vit_transform.stride ** 2)
        vae_tokens = vae_h * vae_w // (self.transform.stride ** 2) // 4  # latent_patch_size=2
        return len(ids) + vit_tokens + vae_tokens

    def _build_full(self, sample):
        msgs = sample['messages']
        prompt = msgs[0]['content']
        asst = msgs[1]['content']
        code = next(c['text'] for c in asst if c['type'] == 'code')
        img_path = next(c['path'] for c in asst if c['type'] == 'image')

        img = self._load_image(img_path)
        vae_t = self.transform(img)
        vit_t = self.vit_transform(img)

        prompt_ids = self.tokenizer.encode(prompt)
        code_ids = self.tokenizer.encode(code)

        image_tensor_list = [vit_t, vae_t]
        text_ids_list = [prompt_ids, code_ids]
        sequence_plan = [
            _TEXT_PLAN(loss=0, is_code=False, enable_cfg=0),  # prompt: no loss, no cfg dropout
            _TEXT_PLAN(loss=1, is_code=True, enable_cfg=1),   # code: with loss
            _VIT_PLAN(enable_cfg=1),
            _VAE_PLAN(loss=1, enable_cfg=1),
        ]
        num_tokens = (len(prompt_ids) + self._token_count(code_ids, vit_t, vae_t))
        return image_tensor_list, text_ids_list, sequence_plan, num_tokens

    def _build_prompt_free(self, sample):
        asst = sample['messages'][0]['content']
        code = next(c['text'] for c in asst if c['type'] == 'code')
        img_path = next(c['path'] for c in asst if c['type'] == 'image')

        img = self._load_image(img_path)
        vae_t = self.transform(img)
        vit_t = self.vit_transform(img)
        code_ids = self.tokenizer.encode(code)

        direction = self.direction
        if direction == 'mixed':
            direction = random.choice(['code2image', 'image2code'])

        if direction == 'code2image':
            image_tensor_list = [vit_t, vae_t]
            text_ids_list = [code_ids]
            sequence_plan = [
                _TEXT_PLAN(loss=1, is_code=True, enable_cfg=0),
                _VIT_PLAN(enable_cfg=0),
                _VAE_PLAN(loss=1, enable_cfg=0),
            ]
        else:  # image2code
            image_tensor_list = [vit_t, vae_t]
            text_ids_list = [code_ids]
            sequence_plan = [
                _VIT_PLAN(enable_cfg=0),
                _VAE_PLAN(loss=0, enable_cfg=0),  # image is input: no gen loss
                _TEXT_PLAN(loss=1, is_code=True, enable_cfg=0),
            ]

        num_tokens = self._token_count(code_ids, vit_t, vae_t)
        return image_tensor_list, text_ids_list, sequence_plan, num_tokens

    def _build_trace(self, sample):
        raw_steps = sample.get('steps')
        if raw_steps is None:
            return None
        steps = json.loads(raw_steps) if isinstance(raw_steps, str) else raw_steps
        img_path = sample.get('final_image') or sample.get('image_path')
        if not steps or not img_path:
            return None

        steps = steps[:self.max_trace_steps]
        img = self._load_image(img_path)
        vae_t = self.transform(img)
        vit_t = self.vit_transform(img)

        image_tensor_list = []
        text_ids_list = []
        sequence_plan = []
        num_tokens = 0

        for i, step_code in enumerate(steps):
            ids = self.tokenizer.encode(step_code)
            text_ids_list.append(ids)
            num_tokens += len(ids)
            sequence_plan.append(_TEXT_PLAN(loss=1, is_code=True, enable_cfg=1))

            is_final = (i == len(steps) - 1)
            image_tensor_list.append(vit_t.clone() if not is_final else vit_t)
            sequence_plan.append(_VIT_PLAN(enable_cfg=1))
            if is_final:
                image_tensor_list.append(vae_t)
                sequence_plan.append(_VAE_PLAN(loss=1, enable_cfg=1))
            # Intermediate steps: skip VAE (no intermediate render available)
            # TODO: replace with pre-rendered intermediate images for proper trace training

        vit_h, vit_w = vit_t.shape[1], vit_t.shape[2]
        vae_h, vae_w = vae_t.shape[1], vae_t.shape[2]
        num_tokens += (vit_h * vit_w // (self.vit_transform.stride ** 2)) * len(steps)
        num_tokens += vae_h * vae_w // (self.transform.stride ** 2) // 4
        return image_tensor_list, text_ids_list, sequence_plan, num_tokens

    def __iter__(self):
        data_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start = self.data_status.get(worker_id, 0) + 1
        else:
            row_start = 0

        while True:
            for row_idx, (json_str, orig_idx) in enumerate(data_per_worker[row_start:],
                                                            start=row_start):
                try:
                    sample = json.loads(json_str)
                    seq_type = sample.get('type', 'full')

                    if seq_type == 'full':
                        result = self._build_full(sample)
                    elif seq_type == 'prompt_free':
                        result = self._build_prompt_free(sample)
                    elif seq_type == 'trace':
                        result = self._build_trace(sample)
                    else:
                        continue

                    if result is None:
                        continue

                    image_tensor_list, text_ids_list, sequence_plan, num_tokens = result

                    has_loss = any(item.get('loss', 0) for item in sequence_plan)
                    if not has_loss:
                        continue

                    yield dict(
                        image_tensor_list=image_tensor_list,
                        text_ids_list=text_ids_list,
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": orig_idx,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )
                except Exception:
                    traceback.print_exc()
                    continue

            row_start = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
