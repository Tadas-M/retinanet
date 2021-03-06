import argparse
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import OpenImagesDataset, Resizer, Normalizer
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--oi_path', help='Path to Open Images')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--oi_classes', help='Path to classlist csv', type=str)
    parser.add_argument('--iou_threshold', help='IOU threshold used for evaluation', type=str, default='0.5')
    parser = parser.parse_args(args)

    dataset_val = OpenImagesDataset(root_dir=parser.oi_path,
                                    data_dir="test",
                                    class_file_path="metadata/class-descriptions-boxable.csv",
                                    class_list=parser.oi_classes,
                                    transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    retinanet = torch.load(parser.model_path, map_location=map_location)  #model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
        retinanet.load_state_dict(torch.load(parser.model_path, map_location=map_location).module.state_dict())
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        map_location = 'cpu'
        #retinanet.load_state_dict(torch.load(parser.model_path, map_location=map_location))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    print(csv_eval.evaluate(dataset_val, retinanet, iou_threshold=float(parser.iou_threshold)))


if __name__ == '__main__':
    main()
