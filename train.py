import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import create_dataloader, denormalize
from model import AdainStyleTransfom

def train(opt):
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
    (ROOT / 'weights').mkdir(parents=True, exist_ok=True)

    # preparae dataset
    train_dataloader = create_dataloader(opt.train_content_dir, opt.train_style_dir, opt.img_size, opt.crop_size, opt.batch_size, num_workers=opt.workers)
    
    # create model
    model = AdainStyleTransfom()

    # resume
    if opt.resume:
        model.load_state_dict(torch.load(opt.resume))

    if opt.gpu != "cpu" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=opt.lr)

    # training
    for i in range(opt.epochs):
        model.train()
        print(f"Epoch: {i+1}/{opt.epochs}")

        for content, style in tqdm(train_dataloader):
            optimizer.zero_grad()

            content = content.to(device)
            style = style.to(device)

            loss_c, loss_s = model(content, style)
            loss = opt.content_weight * loss_c.mean() + opt.style_weight * loss_s.mean()
            
            loss.backward()
            optimizer.step()

            tqdm.write(f"content loss: {loss_c.item():.5f} \t style loss: {loss_s.item():.5f} \t total loss: {loss:.5f}")
            
        # snapshot
        model.eval()
        with torch.no_grad():
            content_test, style_test = next(iter(train_dataloader))

            content_test = content_test.to(device)
            style_test = style_test.to(device)

            if isinstance(model, nn.DataParallel):
                output = model.module.predict(content_test, style_test)
            else:
                output = model.predict(content_test, style_test)

            content_test = denormalize(content_test)
            style_test = denormalize(style_test)
            output = denormalize(output)

            res = torch.cat([content_test, style_test, output], dim=0)
            save_image(res, f'{str(ROOT)}/epoch_{i+1}_iteration.png', nrow=opt.batch_size)

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), f'{ROOT}/weights/epoch_{i+1}.pth')
        else:
            torch.save(model.state_dict(), f'{ROOT}/weights/epoch_{i+1}.pth')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # basic setting
    parser.add_argument("--train-content", type=str, required=True, help="training content path")
    parser.add_argument("--train-style", type=str, required=True, help="training style path")
    parser.add_argument("--save-dir", type=str, default=f"./log/{datetime.now().strftime('%Y-%m-%d-%H-%M')}")

    # training setting
    parser.add_argument("--gpu", default="0", help='i.e. 0 or 0,2 or cpu')
    parser.add_argument('--workers', type=int, default=0, help="dataloader workers")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--content-weight', type=float, default=1.0)
    parser.add_argument('--style-weight', type=float, default=10.0)
    parser.add_argument('--img-size', type=int, default=512, help="resize images")
    parser.add_argument('--crop-size', type=int, default=256, help="random crop size")
    parser.add_argument('--resume', type=str, help="resume training, weights path required")
    opt = parser.parse_args()
    print(opt)

    train(opt)
