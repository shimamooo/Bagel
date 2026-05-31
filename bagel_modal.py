import os
import sys
import modal

GPU        = "A100-40GB"
MODE       = 1
MAX_MEM_GB = 38

app = modal.App("bagel")

MODEL_DIR = "/vol/models/BAGEL-7B-MoT"
REPO_ID   = "ByteDance-Seed/BAGEL-7B-MoT"

model_volume = modal.Volume.from_name("bagel-model-weights", create_if_missing=True)

bagel_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libgl1", "libglib2.0-0", "git", "clang")
    .pip_install(
        "numpy==1.24.4",
        "scipy==1.10.1",
        "Pillow",
        "opencv-python-headless",
        "decord==0.6.0",
        "safetensors==0.4.5",
        "PyYAML==6.0.2",
        "requests",
        "pyarrow==11.0.0",
        "huggingface_hub==0.29.1",
        "sentencepiece==0.1.99",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.49.0",
        "einops==0.8.1",
        "accelerate>=0.34.0",
        "bitsandbytes",
        "triton",
        "ninja",
        "wheel",
        "setuptools",
        "packaging",
        "gradio>=4.0",
        "matplotlib==3.7.0",
        "xlsxwriter",
        "wandb",
    )
    .run_commands("pip install flash-attn==2.5.8 --no-build-isolation")
    .add_local_dir(
        ".",
        remote_path="/app",
        copy=False,
        ignore=[".git", "models", "offload", "**/__pycache__", "**/*.pyc",
                ".pytest_cache", "*.egg-info"],
    )
)


@app.function(
    image=bagel_image,
    volumes={"/vol": model_volume},
    timeout=7200,
    cpu=8,
)
def download_weights():
    """Download BAGEL-7B-MoT weights to the persistent Modal Volume.

    Run once:
        uv run modal run bagel_modal.py::download_weights
    """
    from huggingface_hub import snapshot_download

    ema_path = os.path.join(MODEL_DIR, "ema.safetensors")
    if os.path.exists(ema_path):
        print(f"Model already cached at {MODEL_DIR} — nothing to do.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading {REPO_ID} → {MODEL_DIR}  (≈28 GB, may take a while)")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
    )
    model_volume.commit()
    print("Download complete.")


@app.cls(
    image=bagel_image,
    gpu=GPU,
    volumes={"/vol": model_volume},
    timeout=3600,
    max_containers=1,
    scaledown_window=300,
    startup_timeout=900,
    env={"GRADIO_TEMP_DIR": "/tmp/gradio"},
)
class BagelApp:
    """Loads BAGEL once on container start, then serves the Gradio UI."""

    @modal.enter()
    def load(self):
        import time, random
        import numpy as np
        import torch
        from PIL import Image
        from accelerate import (infer_auto_device_map,
                                load_checkpoint_and_dispatch,
                                init_empty_weights)

        os.makedirs("/tmp/gradio", exist_ok=True)
        os.environ["GRADIO_TEMP_DIR"] = "/tmp/gradio"

        import gradio as gr

        sys.path.insert(0, "/app")
        os.chdir("/app")

        t0 = time.time()
        def log(msg):
            print(f"[+{time.time()-t0:5.0f}s] {msg}", flush=True)

        log("importing BAGEL modules")
        from data.data_utils import add_special_tokens, pil_img2rgb
        from data.transforms import ImageTransform
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer
        from inferencer import InterleaveInferencer

        log("reading model configs")
        llm_config = Qwen2Config.from_json_file(f"{MODEL_DIR}/llm_config.json")
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(f"{MODEL_DIR}/vit_config.json")
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        log("loading VAE (ae.safetensors)")
        vae_model, vae_config = load_ae(local_path=f"{MODEL_DIR}/ae.safetensors")
        log("VAE loaded")

        log("building empty model skeleton")
        config = BagelConfig(
            visual_gen=True, visual_und=True,
            llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
            latent_patch_size=2, max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model      = SiglipVisionModel(vit_config)
            model          = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True)
        log("skeleton ready")

        log("loading tokenizer")
        tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_DIR)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        log(f"computing device map  (capping at {MAX_MEM_GB} GiB per GPU)")
        device_map = infer_auto_device_map(
            model,
            max_memory={i: f"{MAX_MEM_GB}GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device_modules = [
            "language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
            "vae2llm", "llm2vae", "connector", "vit_pos_embed",
        ]
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = first_device if k in device_map else "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
        n_gpu  = sum(1 for v in device_map.values() if str(v).startswith("cuda"))
        n_cpu  = sum(1 for v in device_map.values() if v == "cpu")
        n_disk = sum(1 for v in device_map.values() if v == "disk")
        log(f"device map: {n_gpu} GPU layers | {n_cpu} CPU layers | {n_disk} disk layers")
        if n_cpu or n_disk:
            log("offloaded layers detected - inference will be slow. "
                "Increase MAX_MEM_GB or use a larger GPU.")

        # ── load weights ──────────────────────────────────────────────
        log("dispatching ema.safetensors, GPU  (typically 2-4 min)")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=f"{MODEL_DIR}/ema.safetensors",
            device_map=device_map,
            offload_buffers=True,
            offload_folder="/tmp/offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
        vram = torch.cuda.memory_allocated() / 1e9
        log(f"weights loaded, VRAM in use: {vram:.1f} GB / {MAX_MEM_GB} GB")

        log("building inferencer")
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14),
            new_token_ids=new_token_ids,
        )

        def set_seed(seed):
            if seed > 0:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        RATIOS = {"1:1":(1024,1024),"4:3":(768,1024),"3:4":(1024,768),
                  "16:9":(576,1024),"9:16":(1024,576)}

        def text_to_image(prompt, thinking, cfg_text, cfg_interval, ts_shift,
                          n_steps, renorm_min, renorm_type,
                          max_think, do_sample, temp, seed, ratio):
            set_seed(seed)
            result = inferencer(
                text=prompt, think=thinking,
                max_think_token_n=max_think if thinking else 1024,
                do_sample=do_sample if thinking else False,
                text_temperature=temp if thinking else 0.3,
                cfg_text_scale=cfg_text, cfg_interval=[cfg_interval, 1.0],
                timestep_shift=ts_shift, num_timesteps=n_steps,
                cfg_renorm_min=renorm_min, cfg_renorm_type=renorm_type,
                image_shapes=RATIOS.get(ratio, (1024, 1024)),
            )
            return result["image"], result.get("text") or ""

        def edit_image(image, prompt, thinking, cfg_text, cfg_img, cfg_interval,
                       ts_shift, n_steps, renorm_min, renorm_type,
                       max_think, do_sample, temp, seed):
            if image is None:
                return None, "Upload an image first."
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            set_seed(seed)
            result = inferencer(
                image=pil_img2rgb(image), text=prompt, think=thinking,
                max_think_token_n=max_think if thinking else 1024,
                do_sample=do_sample if thinking else False,
                text_temperature=temp if thinking else 0.3,
                cfg_text_scale=cfg_text, cfg_img_scale=cfg_img,
                cfg_interval=[cfg_interval, 1.0],
                timestep_shift=ts_shift, num_timesteps=n_steps,
                cfg_renorm_min=renorm_min, cfg_renorm_type=renorm_type,
            )
            return result["image"], result.get("text") or ""

        def understand(image, prompt, thinking, do_sample, temp, max_tokens):
            if image is None:
                return "Upload an image first."
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            result = inferencer(
                image=pil_img2rgb(image), text=prompt, think=thinking,
                understanding_output=True,
                do_sample=do_sample, text_temperature=temp,
                max_think_token_n=max_tokens,
            )
            return result["text"]

        log("building Gradio UI")

        def hyper_row1(cfg_text_default=4.0, cfg_interval_default=0.4):
            with gr.Row():
                cfg_text     = gr.Slider(1.0, 8.0,  value=cfg_text_default, step=0.1, label="CFG text scale")
                cfg_interval = gr.Slider(0.0, 1.0,  value=cfg_interval_default, step=0.05, label="CFG start interval")
            return cfg_text, cfg_interval

        def think_widgets():
            max_think  = gr.Slider(64, 4096, value=1024, step=64,  label="Max think tokens",  visible=False)
            do_sample  = gr.Checkbox(label="Sampling", value=False, visible=False)
            temp       = gr.Slider(0.1, 1.0, value=0.3, step=0.1,  label="Temperature",       visible=False)
            return max_think, do_sample, temp

        with gr.Blocks(title="BAGEL") as demo:
            gr.Markdown("# BAGEL Multimodal Understanding & Generation")

            with gr.Tab("Text to Image"):
                t2i_prompt = gr.Textbox(label="Prompt", lines=3,
                    value="A female cosplayer portraying an ethereal fairy or elf, "
                          "wearing a flowing dress in soft mystical colors like emerald "
                          "green and silver, magical forest background.")
                t2i_think = gr.Checkbox(label="Thinking mode", value=False)
                with gr.Accordion("Hyperparameters", open=False):
                    with gr.Row():
                        t2i_seed  = gr.Slider(0, 1_000_000, value=0, step=1, label="Seed (0=random)")
                        t2i_ratio = gr.Dropdown(list(RATIOS), value="1:1", label="Aspect ratio")
                    t2i_cfg_text, t2i_cfg_int = hyper_row1()
                    with gr.Row():
                        t2i_renorm_type = gr.Dropdown(["global","channel","text_channel"],
                                                      value="global", label="CFG renorm type")
                        t2i_renorm_min  = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="CFG renorm min")
                    with gr.Row():
                        t2i_steps    = gr.Slider(10, 100, value=50, step=5,  label="Timesteps")
                        t2i_ts_shift = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Timestep shift")
                    t2i_max_think, t2i_do_sample, t2i_temp = think_widgets()
                t2i_think.change(lambda v: [gr.update(visible=v)]*3,
                                 t2i_think, [t2i_max_think, t2i_do_sample, t2i_temp])
                t2i_btn     = gr.Button("Generate", variant="primary")
                t2i_out     = gr.Image(label="Generated image")
                t2i_thought = gr.Textbox(label="Thinking process", visible=False)
                t2i_think.change(lambda v: gr.update(visible=v), t2i_think, t2i_thought)
                gr.on([t2i_btn.click, t2i_prompt.submit], fn=text_to_image,
                      inputs=[t2i_prompt, t2i_think, t2i_cfg_text, t2i_cfg_int,
                               t2i_ts_shift, t2i_steps, t2i_renorm_min, t2i_renorm_type,
                               t2i_max_think, t2i_do_sample, t2i_temp, t2i_seed, t2i_ratio],
                      outputs=[t2i_out, t2i_thought])

            with gr.Tab("🖌️ Image Edit"):
                with gr.Row():
                    with gr.Column():
                        ed_img_in  = gr.Image(label="Input image")
                        ed_prompt  = gr.Textbox(label="Edit instruction", lines=2,
                            value="She boards a modern subway, quietly reading a folded newspaper, "
                                  "wearing the same clothes.")
                    with gr.Column():
                        ed_img_out  = gr.Image(label="Edited image")
                        ed_thought  = gr.Textbox(label="Thinking process", visible=False)
                ed_think = gr.Checkbox(label="Thinking mode", value=False)
                with gr.Accordion("Hyperparameters", open=False):
                    with gr.Row():
                        ed_seed    = gr.Slider(0, 1_000_000, value=0, step=1, label="Seed")
                        ed_cfg_img = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="CFG image scale")
                    ed_cfg_text, ed_cfg_int = hyper_row1(cfg_text_default=4.0,
                                                          cfg_interval_default=0.0)
                    with gr.Row():
                        ed_renorm_type = gr.Dropdown(["global","channel","text_channel"],
                                                     value="text_channel", label="CFG renorm type")
                        ed_renorm_min  = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="CFG renorm min")
                    with gr.Row():
                        ed_steps    = gr.Slider(10, 100, value=50, step=5,   label="Timesteps")
                        ed_ts_shift = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Timestep shift")
                    ed_max_think, ed_do_sample, ed_temp = think_widgets()
                ed_think.change(lambda v: [gr.update(visible=v)]*3,
                                ed_think, [ed_max_think, ed_do_sample, ed_temp])
                ed_think.change(lambda v: gr.update(visible=v), ed_think, ed_thought)
                ed_btn = gr.Button("Edit", variant="primary")
                gr.on([ed_btn.click, ed_prompt.submit], fn=edit_image,
                      inputs=[ed_img_in, ed_prompt, ed_think, ed_cfg_text, ed_cfg_img,
                               ed_cfg_int, ed_ts_shift, ed_steps, ed_renorm_min,
                               ed_renorm_type, ed_max_think, ed_do_sample, ed_temp, ed_seed],
                      outputs=[ed_img_out, ed_thought])

            with gr.Tab("Image Understanding"):
                with gr.Row():
                    with gr.Column():
                        und_img    = gr.Image(label="Input image")
                        und_prompt = gr.Textbox(label="Question", lines=2,
                            value="Can someone explain what's funny about this meme??")
                    und_out = gr.Textbox(label="Answer", lines=20)
                und_think = gr.Checkbox(label="Thinking mode", value=False)
                with gr.Accordion("Hyperparameters", open=False):
                    with gr.Row():
                        und_sample   = gr.Checkbox(label="Sampling", value=False)
                        und_temp     = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
                        und_max_tok  = gr.Slider(64, 4096, value=512, step=64, label="Max tokens")
                und_btn = gr.Button("Submit", variant="primary")
                gr.on([und_btn.click, und_prompt.submit], fn=understand,
                      inputs=[und_img, und_prompt, und_think,
                               und_sample, und_temp, und_max_tok],
                      outputs=und_out)

        self._demo = demo
        self._demo.queue()
        log(f"BAGEL ready, total load time: {time.time()-t0:.0f}s")

    @modal.asgi_app()
    def ui(self):
        """Return the Gradio Blocks as an ASGI app (used by `modal serve`)."""
        return self._demo.app
