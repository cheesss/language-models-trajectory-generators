import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import config
from openai import OpenAI
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from dotenv import load_dotenv
import os
import json
import multiprocessing
import main

sys.path.append("./XMem/")
load_dotenv("openaiAPI.env")
api_key = os.getenv("api_key")

import logging

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,  # 출력할 최소 레벨 설정
    format='%(asctime)s - %(levelname)s - %(message)s',  # 출력 형식 설정
    handlers=[logging.StreamHandler()]  # 콘솔 출력 핸들러
)
logger = logging.getLogger(__name__)



logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)



from XMem.inference.inference_core import InferenceCore
from XMem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

def get_langsam_output(image, model, segmentation_texts, segmentation_count):
    # print("segmentation_texts:", segmentation_texts)
    # print(type(segmentation_texts))

    segmentation_texts = " . ".join(segmentation_texts)

    # masks, boxes, phrases, logits = model.predict(image, segmentation_texts)
    data= model.predict(images_pil=image, texts_prompt=segmentation_texts)
    output_file_txt = "model_output.txt"
    with open(output_file_txt, "w") as f:
        f.write(str(data))



    result_dict = data 
    # print("result_dict=",result_dict)

    logits = [item['scores'] for item in result_dict]
    phrases = [item['labels'] for item in result_dict]
    boxes = [item['boxes'] for item in result_dict]
    masks = [item['masks'] for item in result_dict]

    # logger.info("boxes length = "+ str(len(boxes)))

    output_file = "output_data.txt"
    with open(output_file, "w") as f:
        f.write(f"logits: {logits}\n")
        f.write(f"phrases: {phrases}\n")
        f.write(f"boxes: {boxes}\n")
        f.write(f"masks: {masks}\n")
    print(np.shape(masks))
    _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
    [a.axis("off") for a in ax.flatten()]
    ax[0].imshow(image)



    count = sum(len(sublist) for sublist in boxes)
    colors1 = []
    colors2 = []
    for i in range(count):
        colors1.append("red")
        colors2.append("cyan")
    # logger.info(" = "+str(count))
    # logger.info("boxes colors = "+str(colors1)+str(colors2))



    for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
        to_tensor = transforms.PILToTensor()
        image_tensor = to_tensor(image)
        box = torch.tensor(box)
        # logger.info(box.shape)
        # box = box.unsqueeze(dim=0)
        # logger.info(box.shape)
        # 물체 개수가 변하면 아래 색깔 개수를 바꿔줘야한다.
        image_tensor = draw_bounding_boxes(image_tensor, box, colors=colors1, width=3)
        mask = torch.tensor(mask)
        mask = mask.bool()
        image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=colors2)
        to_pil_image = transforms.ToPILImage()
        image_pil = to_pil_image(image_tensor)

        ax[1 + i].imshow(image_pil)
        ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":"white", "edgecolor":"red", "boxstyle":"square"})


    plt.savefig(config.langsam_image_path.format(object=segmentation_count))
    plt.show()

    masks = torch.tensor(masks)
    masks = masks.float()

    return masks, boxes, phrases



def get_chatgpt_output(model, new_prompt, messages, role, file=sys.stdout):

    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role":role, "content":new_prompt})
    # llm에게 전달해줄 메세지를 편집해준다.

    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""
    # 변수 초기화

    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)

    messages.append({"role":"assistant", "content":new_output})
    main.save_code_block_to_file(messages, file_name="message_blocks.txt")

    return messages



def get_xmem_output(model, device, trajectory_length):

    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
    mask = np.unique(mask, return_inverse=True)[1].reshape(mask.shape)
    num_objects = len(np.unique(mask)) - 1

    torch.cuda.empty_cache()

    processor = InferenceCore(model, config.xmem_config)
    processor.set_all_labels(range(1, num_objects + 1))

    masks = []

    with torch.cuda.amp.autocast(enabled=True):

        for i in range(0, trajectory_length + 1, config.xmem_output_every):

            frame = np.array(Image.open(config.rgb_image_head_path).convert("RGB"))

            frame_torch, _ = image_to_torch(frame, device)
            if i == 0:
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                prediction = processor.step(frame_torch)

            prediction = torch_prob_to_numpy_mask(prediction)
            masks.append(prediction)

            if i % config.xmem_visualise_every == 0:
                visualisation = overlay_davis(frame, prediction)
                output = Image.fromarray(visualisation)
                output.save(config.xmem_output_path.format(step=i))

    return masks