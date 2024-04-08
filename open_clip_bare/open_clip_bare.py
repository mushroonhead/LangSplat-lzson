#!/usr/bin/env python
import torch
import torchvision
import open_clip


class OpenClipBarebones(torch.nn.Module):
    """
    A barebone implementation of the open clip network, 
    without the extra processing found in original OpenCLIPNetwork by langsplat group
    """
    def __init__(self, device: torch.device, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)
        self.device = device

    def forward(self, text_list):
        return self.encode_text(text_list)

    def encode_image(self, input, mask=None):
        processed_input = self.process(input).half().to(self.device)
        return self.model.encode_image(processed_input, mask=mask)

    def encode_text(self, text_list):
        text = self.tokenizer(text_list).to(self.device)
        return self.model.encode_text(text)