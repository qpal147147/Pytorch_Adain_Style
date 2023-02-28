import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision.utils import save_image

from dataloader import create_dataloader ,denormalize
from model import AdainStyleTransfom

def test(opt):
    # set cpu or gpu
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
        device = "cuda:0"
    elif opt.gpu == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = -1
        device = "cpu"

    # create training directory
    ROOT = Path(opt.save_dir)
    ROOT.mkdir(parents=True, exist_ok=True)

    # preparae dataset
    dataloader = create_dataloader(opt.content, opt.style, batch_size=1, shuffle=False, num_workers=0, training=False)
    
    # get filename
    content_p = Path(opt.content)
    style_p = Path(opt.style)

    if content_p.is_dir() and style_p.is_dir():
        content_names = [x.stem for x in content_p.glob("*.*") if x.is_file()]
        style_names = [x.stem for x in style_p.glob("*.*") if x.is_file()]
    elif content_p.is_file() and style_p.is_file():
        content_names = [content_p.stem]
        style_names = [style_p.stem]

    # load weights
    model = AdainStyleTransfom(opt.alpha)
    model.load_state_dict(torch.load(opt.weights))
    model = model.to(device)
    model.eval()

    # inference
    for i, (content, style) in enumerate(tqdm(dataloader)):
        content = content.to(device)
        style = style.to(device)

        res = model.predict(content, style)
        res = denormalize(res)
        save_image(res, f'{str(ROOT)}/{content_names[i]}_{style_names[i]}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, required=True, help="weights path")
    parser.add_argument("--content", type=str, required=True, help="test content directory or file")
    parser.add_argument("--style", type=str, required=True, help="test style directory or file")
    parser.add_argument('--alpha', type=float, default=1.0, help="control generate image style, 0~1")
    parser.add_argument("--gpu", default="0", help='i.e. 0 or 0,2 or cpu')
    parser.add_argument("--save-dir", type=str, default=f"./test/{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    opt = parser.parse_args()
    print(opt)

    test(opt)