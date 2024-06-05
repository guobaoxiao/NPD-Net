import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
# from lib.NPD_Net import NPD_Net
from lib.NPD_Net import NPD_Net   # change backbone from res2et to pvt
from lib.isnet import ISNetGTEncoder
from utilss.dataloader import get_loader
from utilss.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import torch.nn as nn
train_loss = []

fea_loss = nn.MSELoss(size_average=True)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def structure_loss_for_feature(pred, mask):
    loss = 0.0
    for i in range(0, len(pred)):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask[i], kernel_size=31, stride=1, padding=15) - mask[i])
        pred[i] = _upsample_like(pred[i], mask[i])
        wbce = F.binary_cross_entropy_with_logits(pred[i], mask[i], reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred[i] = torch.sigmoid(pred[i])
        inter = ((pred[i] * mask[i]) * weit).sum(dim=(2, 3))
        union = ((pred[i] + mask[i]) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        loss = loss + (wbce + wiou).mean()
    return loss / len(pred)

def muti_loss_fusion_kl(dfs, fs, mode='MSE'):

    loss = 0.0
    for i in range(0, len(dfs)):
        if (mode == 'MSE'):
            dfs[i] = _upsample_like(dfs[i], fs[i])
            loss = loss + fea_loss(dfs[i], fs[i])  ### add the mse loss of features as additional constraints
    return loss

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src

def get_gt_encoder(train_loader, model, epoch, save_path):
    optimizer_gt = torch.optim.Adam(model.parameters(), opt.lr)

    if(opt.gt_encoder_model!=""):
        model_path = opt.model_path+"/"+opt.gt_encoder_model
        if torch.cuda.is_available():
            model.module.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path,map_location="cpu"))
        print("gt encoder restored from the saved weights ...")
        return model
    for epoch in range(1, epoch):
        model.train()
        # ---- multi-scale training ----
        size_rates = [0.75, 1, 1.25]
        loss_record6, loss_record5,loss_record4, loss_record3, loss_record2, loss_record1 = AvgMeter(), AvgMeter(),  AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer_gt.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                [lateral_map_6,lateral_map_5,lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1]\
                    ,[gtmidf1, gtmidf2, gtmidf3,gtmidf4, gtmidf5, gtmidf5] = model(gts)
                # ---- loss function ----
                loss6 = structure_loss(lateral_map_6, gts)
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss1 = structure_loss(lateral_map_1, gts)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6   # TODO: try different weights for loss

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer_gt, opt.clip)
                optimizer_gt.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record1.update(loss1.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
                    loss_record5.update(loss5.data, opt.batchsize)
                    loss_record6.update(loss6.data, opt.batchsize)
            # ---- train visualization ----
            if i % 5 == 0 or i == total_step:
                print('GTENCODER-{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      'lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f},lateral-5: {:0.4f}, lateral-6: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(),loss_record5.show(), loss_record6.show()))
                train_loss.append(loss_record1.show().item())
                train_loss.append(loss_record2.show().item())
                train_loss.append(loss_record3.show().item())
                train_loss.append(loss_record4.show().item())
                train_loss.append(loss_record5.show().item())
                train_loss.append(loss_record6.show().item())

                with open("./train_loss.txt", 'w') as train_los:
                    train_los.write(str(train_loss))
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}] '.
                          format(datetime.now(), epoch, opt.epoch, i, total_step) + ": loss save Successfully!")
                    train_los.close()

        if (epoch + 1) % 2 == 0:
            torch.save(model.module.state_dict(), save_path + "loss-" +str(round(loss.item(), 4)) +'-GTENCODER-NPD_Net-%d.pth' % epoch)
            print('GTENCODER-[Saving Snapshot:]', save_path + 'GTENCODER-NPD_Net-%d.pth' % epoch)
            if epoch > 18:    # 当 gtencoder 的epoch大于18时 训练完成
                return model

def train(train_loader, model, optimizer, epoch, save_path_for_weigth):

    print("-" * 20, "Start Train featurenet", "-" * 20)
    if opt.interm_sup:
        # print("Get the gt encoder ...")
        gt_model = torch.nn.DataParallel(ISNetGTEncoder(), device_ids=[0, 1]).cuda()
        featurenet = get_gt_encoder(train_loader, gt_model, opt.epoch, save_path_for_weigth)
        ## freeze the weights of gt encoder
        for param in featurenet.parameters():
            param.requires_grad = False

    for epoch in range(1, epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        print("-"*20 + "start train" + "-"*20)
        model.train()
        # ---- multi-scale training ----
        size_rates = [0.75, 1, 1.25]
        loss_structure, loss_feature, loss_record6, loss_record5, loss_record4, loss_record3, loss_record2, loss_record1 = AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                ds, mds = model(images)
                _, gtmsd = featurenet(gts)
                # ---- loss function ----
                loss6 = structure_loss(ds[0], gts)
                loss5 = structure_loss(ds[1], gts)
                loss4 = structure_loss(ds[2], gts)
                loss3 = structure_loss(ds[3], gts)
                loss2 = structure_loss(ds[4], gts)
                loss1 = structure_loss(ds[5], gts)
                # feature_loss = structure_loss_for_feature(mds, gtmsd)
                feature_loss = muti_loss_fusion_kl(mds, gtmsd, mode='MSE')
                # print("feature loss add ----")
                strcution_loss_fi = (1 - opt.a) * (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)
                feature_loss_fi = opt.a * feature_loss
                loss = strcution_loss_fi + feature_loss_fi

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record1.update(loss1.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
                    loss_record5.update(loss5.data, opt.batchsize)
                    loss_record6.update(loss6.data, opt.batchsize)
                    loss_structure.update(strcution_loss_fi.data, opt.batchsize)
                    loss_feature.update(feature_loss_fi.data, opt.batchsize)
            # ---- train visualization ----
            if i % 100 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      'lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}lateral-5: {:0.4f}, lateral-6: {:0.4f}, feature loss: {:0.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show(), loss_record6.show(),loss_feature.show()))
                train_loss.append(loss_feature.show().item())
                train_loss.append(loss_structure.show().item())
                with open("./train_loss.txt", 'w') as train_los:
                    train_los.write(str(train_loss))
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}] '.
                          format(datetime.now(), epoch, opt.epoch, i, total_step) + ": loss save Successfully!")
                    train_los.close()

        if (epoch) % 1 == 0:
            torch.save(model.module.state_dict(), save_path_for_weigth + "struction_loss: " + str(round(loss.item()-feature_loss_fi.item(), 4))+ "feature_loss: "
                       + str(round(feature_loss_fi.item(), 4)) + 'NPD_Net-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path_for_weigth + "struction_loss: " + str(round(loss.item()-feature_loss_fi.item(), 4)) + "feature_loss: "
                       + str(round(feature_loss_fi.item(), 4)) + 'NPD_Net-%d.pth' % epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='data/TrainDataset')
    parser.add_argument('--train_save', type=str, default='NPD_Net')
    parser.add_argument("--interm_sup", type=bool, default=True)
    parser.add_argument("--gt_encoder_model",type=str,default='loss-10.5362-GTENCODER-NPD_Net-1.pth')
    parser.add_argument("--model_path", type=str, default='weight/after-train/')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--a', type=int, default=0.0905)


    opt = parser.parse_args()


    model = torch.nn.DataParallel(NPD_Net(), device_ids=[0, 1]).cuda()

    params = model.parameters()

    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    train(train_loader, model, optimizer, opt.epoch, opt.model_path)
