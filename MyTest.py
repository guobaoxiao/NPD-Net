import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.NPD_Net import NPD_Net
from utilss.dataloader import test_dataset
import imageio
from skimage import img_as_ubyte


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352)
parser.add_argument('--pth_path', type=str, default='/data/Jenny/NPD-Net/weight/after-train/xxx.pth'
                                                    )

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = '/data/Jenny/NPD-Net/data/TestDataset/'
    save_path = '/data/Jenny/NPD-Net/weight/after-test/'+_data_name+"/"
    opt = parser.parse_args()
    model = NPD_Net()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # os.makedirs(save_path, exist_ok=True)
    image_root = data_path + _data_name + "/images/"
    gt_root = data_path + _data_name + "/masks/"
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        [res6, res5, res4, res3, res2, res1], _ = model(image)
        res = res2

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # Check if the directory exists
        if not os.path.exists(save_path):
            # If it doesn't exist, create the directory
            os.makedirs(save_path)
            print(f"Directory {save_path} created.")
        imageio.imwrite(save_path+name, img_as_ubyte(res))

