import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from utils.draw_box_utils import draw_box
from utils.model_utils import create_model


class APP(object):
    def __init__(self,
                 backbone,
                 RCNN,
                 model_weights,
                 num_classes,
                 ):
        self.model = create_model(backbone, RCNN, num_classes)
        self.model.cuda()
        checkpoint = torch.load(model_weights, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

        self.class_IDs = {1: 'bus',
                          2: 'truck',
                          3: 'suv',
                          4: 'rickshaw',
                          5: 'car',
                          6: 'motorbike',
                          7: 'three wheelers (CNG)',
                          8: 'van',
                          9: 'pickup',
                          10: 'bicycle',
                          11: 'minivan',
                          12: 'scooter',
                          13: 'human hauler',
                          14: 'army vehicle',
                          15: 'ambulance',
                          16: 'auto rickshaw',
                          17: 'policecar',
                          18: 'taxi',
                          19: 'minibus',
                          20: 'wheelbarrow',
                          21: 'garbagevan'}

    def preprocess(self, img):
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        return img

    def run(self, image_path):
        img = Image.open(image_path)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.preprocess(img).cuda())[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("No target detected!")

            draw_box(img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     self.class_IDs,
                     thresh=0.3,
                     line_thickness=3)
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    class detectorConfig:
        min_size = 800
        max_size = 1000
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        # anchor parameters
        anchor_size = [64, 128, 256]
        anchor_ratio = [0.5, 1, 2.0]

        # roi align parameters
        roi_out_size = [7, 7]
        roi_sample_rate = 2

        # rpn process parameters
        rpn_pre_nms_top_n_train = 2000
        rpn_post_nms_top_n_train = 2000

        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_test = 1000

        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5

        # remove low threshold target
        box_score_thresh = 0.015
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None


    image_path = "/media/raja/6TB/Own_Trials/Dataset/Kaggle1/Datasets/test1/test/Asraf_50_jpg.rf.7026694f0b9f37a6790982295c7e8663.jpg"
    backbone = 'transformer-CvT'
    model_weights = "/media/raja/6TB/Own_Trials/Sources/vehicleDetection/RCNN_Detection/checkpoints/transformer-model-478.pth"
    num_classes = 21 + 1

    app = APP(backbone,
              detectorConfig(),
              model_weights,
              num_classes)
    app.run(image_path)
