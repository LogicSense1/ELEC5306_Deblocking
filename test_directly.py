import argparse
import os
from src.utils import psnr
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as Image
from src.network import *
import cv2


cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='memnet', help='ARCNN or FastARCNN or DnCNN')
    parser.add_argument('--weights_path', type=str, default="/home/wentian/Documents/ELEC5306_Deblock/models/memnet_q40/checkpoint_q40_046_diff1.319997_2800_best.pth.tar")
    parser.add_argument('--image_path', default="/home/wentian/Documents/ELEC5306_Deblock/test_images/im1.jpg", type=str)
    parser.add_argument('--outputs_dir', default="/home/wentian/Documents/ELEC5306_Deblock/test_images", type=str)
    opt = parser.parse_args()
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    if opt.arch == 'ARCNN':
        model = ARCNN()
    elif opt.arch == 'FastARCNN':
        model = FastARCNN()
    elif opt.arch == 'DnCNN':
        model = DnCNN()
    else:
        model = MemNet(3,64,6,6)

    model = nn.DataParallel(model)
    checkpoint = torch.load(opt.weights_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    filename = os.path.basename(opt.image_path).split('.')[0]
    img_input = Image.open(opt.image_path)
    input = transforms.ToTensor()(img_input).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input)
    pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    output = Image.fromarray(pred, mode='RGB')
    output.save(os.path.join(opt.outputs_dir, '{}_{}.png'.format(filename, opt.arch)))
    original = cv2.imread(opt.image_path)
    contrast2 = cv2.imread(os.path.join(opt.outputs_dir, '{}_{}.png'.format(filename, opt.arch)), 1)
    print(f"PSNR value of predication is {psnr(original, contrast2)} dB")