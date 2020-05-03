import numpy as np
import torch
from torch.jit.annotations import Dict, List, Tuple
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss

from utils import get_area, get_intersection_area, is_box_inside, merge_targets_batch, split_targets_batch


def get_FasterRCNN_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


class JointDetector(nn.Module):
    """
    Some code adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/
    Jointly train 2 Faster RCNN Models: 1 for object detection and 1 for part detection
    Both detectors use proposals and features of each other for improved performance of both
        for each obj, fuse with those parts where intersection_area(obj, part) / area(part) >= `fusion_thresh`
        for each part, fuse with those objects where intersection_area(obj, part) / area(part) >= `fusion_thresh`
        NOTE: Currently using fusion_thresh=1.0 for fast computations
    """
    def __init__(self, obj_n_classes, part_n_classes, fusion_thresh=1.0):
        super(JointDetector, self).__init__()
        self.fusion_thresh = fusion_thresh
        
        print('Creating JointDetector(fusion_thresh=%.2f) for %d Object, %d Part classes...' % (self.fusion_thresh, obj_n_classes, part_n_classes))
        self.object_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.part_detector = fasterrcnn_resnet50_fpn(pretrained=True)

        in_features_obj_det = self.object_detector.roi_heads.box_predictor.cls_score.in_features # 1024
        in_features_part_det = self.part_detector.roi_heads.box_predictor.cls_score.in_features # 1024
        in_features = in_features_obj_det + in_features_part_det

        self.object_detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, obj_n_classes)
        self.part_detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, part_n_classes)

        self.transform = self.object_detector.transform # both detectors have save transforms so anyone can be used
    
    def forward(self, images, obj_targets=None, part_targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            {obj/part}_targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            losses in train mode and obj/part detections (tuple) in eval mode
        """
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        if self.training:
            if obj_targets is None or part_targets is None:
                raise ValueError('In training mode, object and part targets should be passed')

            merged_targets, box_counts_1 = merge_targets_batch(obj_targets, part_targets)
            images, merged_targets = self.transform(images, merged_targets) # transform img and targets
            obj_targets, part_targets = split_targets_batch(merged_targets, box_counts_1)
        else:
            images, _ = self.transform(images)
            obj_targets, part_targets = None, None
        
        # get box proposals
        obj_features, obj_proposals, obj_proposal_losses = self.get_features_proposals_losses(self.object_detector, images, obj_targets, '_OBJ')
        part_features, part_proposals, part_proposal_losses = self.get_features_proposals_losses(self.part_detector, images, part_targets, '_PART')

        obj_proposals, obj_labels, obj_regression_targets = self.sample_proposals(self.object_detector, obj_proposals, obj_targets)
        part_proposals, part_labels, part_regression_targets = self.sample_proposals(self.part_detector, part_proposals, part_targets)
        
        # get object and part features that will be combined before using for final classification/regression
        obj_box_features = self.get_box_features(self.object_detector, obj_features, obj_proposals, images.image_sizes)
        part_box_features = self.get_box_features(self.part_detector, part_features, part_proposals, images.image_sizes)
        
        # perform feature fusion of object and part features
        obj_box_features, part_box_features = self.get_fused_obj_part_features(obj_proposals, obj_box_features, part_proposals, part_box_features)
        
        obj_class_logits, obj_box_regression = self.object_detector.roi_heads.box_predictor(obj_box_features)
        part_class_logits, part_box_regression = self.part_detector.roi_heads.box_predictor(part_box_features)

        # get final object and part detections
        obj_detections, obj_det_losses = self.get_detections_losses(self.object_detector, obj_class_logits, obj_box_regression, obj_labels,
            obj_regression_targets, obj_proposals, images.image_sizes, original_image_sizes, '_OBJ')
        part_detections, part_det_losses = self.get_detections_losses(self.part_detector, part_class_logits, part_box_regression, part_labels,
            part_regression_targets, part_proposals, images.image_sizes, original_image_sizes, '_PART')
        
        losses = {}
        losses.update(obj_proposal_losses)
        losses.update(obj_det_losses)
        losses.update(part_proposal_losses)
        losses.update(part_det_losses)

        return losses if self.training else (obj_detections, part_detections)
    
    def get_features_proposals_losses(self, model, images, targets=None, name=''):
        """
        Arguments:
            model (torchvision FasterRCNN model)
            images (ImageList)
            targets (List[Dict])
            name (str): append this name to keys in proposal_losses
        Returns:
            features (Dict[int/str, Tensor]): output features of model.backbone(images)
            proposals (List[Tensor[N, 4]]): box proposals using model's Region Proposal Network
            proposal_losses (Dict)
        """
        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        proposals, proposal_losses = model.rpn(images, features, targets)
        proposal_losses = {k + name: v for k,v in proposal_losses.items()}

        return features, proposals, proposal_losses
    
    def sample_proposals(self, model, proposals, targets=None):
        """
        Sample some proposals (only in train mode) for further training
        Arguments:
            model (torchvision FasterRCNN model)
            proposals (List[Tensor[N, 4]]): all box proposals
            targets (List[Dict])
        Returns:
            proposals (List[Tensor[N', 4]]): sampled box proposals in train mode else no sampling
            labels (List[Tensor[N']]): ground truth class labels (in train mode)
            regression_targets (List[Tensor[N', 4]]): ground truth targets for box regression (in train mode)
        """
        if self.training:
            proposals, _, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
        
        return proposals, labels, regression_targets
    
    def get_box_features(self, model, features, proposals, image_sizes):
        """
        Compute box_features that will be use in model.roi_heads.box_predictor (FastRCNNPredictor)
        Arguments:
            model (torchvision FasterRCNN model)
            features (Dict[int/str, Tensor]): output features of model.backbone(images)
            proposals (List[Tensor[N, 4]]): box proposals
            image_sizes (list[tuple[int, int]]): original shapes of all images
        Returns:
            box_features (Tensor[N_sum, num_features])
                These features are the box_features for all images in the batch concatenated along axis 0
                E.g., if num of proposals in 2 images are 10 and 20, then N_sum will be 30 where first 10 features are for image 1
        """
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
        box_features = model.roi_heads.box_head(box_features)

        return box_features
    
    def get_fused_obj_part_features(self, obj_proposals, obj_box_features, part_proposals, part_box_features):
        """
        Perform feature fusion to get enhanced representation for object and part
            for each obj, fuse with those parts where intersection_area(obj, part) / area(part) >= fusion_thresh
            for each part, fuse with those objects where intersection_area(obj, part) / area(part) >= fusion_thresh
        Arguments:
            {obj/part}_proposals (List[Tensor[N, 4]]): box proposals of images in batch
            {obj/part}_box_features (Tensor[N_sum, num_{obj/part}_features]): box_features for object and part
        Returns:
            fused_obj_box_features (Tensor[N_sum, num_obj_features + num_part_features])
            fused_part_box_features (num_part_features + num_obj_features)
        """
        related_obj_features, related_part_features = [], []

        N_obj_proposals = [p.shape[0] for p in obj_proposals]
        N_part_proposals = [p.shape[0] for p in part_proposals]
        obj_proposal_idx_range = np.cumsum([0] + N_obj_proposals) # to splice out features from obj_box_features for an image
        part_proposal_idx_range = np.cumsum([0] + N_part_proposals) # to splice out features from part_box_features for an image

        batch_size = len(obj_proposals)
        for i in range(batch_size):
            img_obj_proposals = obj_proposals[i]
            img_part_proposals = part_proposals[i]
            
            # Separate out obj_box_features and part_box_features only for current image
            img_obj_box_features = obj_box_features[obj_proposal_idx_range[i]: obj_proposal_idx_range[i+1]]
            img_part_box_features = part_box_features[part_proposal_idx_range[i]: part_proposal_idx_range[i+1]]
            
            for obj_box in img_obj_proposals:
                related_part_feats = self.get_related_part_box_features(obj_box, img_part_proposals, img_part_box_features)
                related_part_feats = related_part_feats.mean(0)
                related_part_features.append(related_part_feats)

            for part_box in img_part_proposals:
                related_obj_feats = self.get_related_obj_box_features(part_box, img_obj_proposals, img_obj_box_features)
                related_obj_feats = related_obj_feats.mean(0)
                related_obj_features.append(related_obj_feats)

        related_obj_features = torch.stack(related_obj_features, dim=0) # [N_part_box_features, in_features_obj_det]
        related_part_features = torch.stack(related_part_features, dim=0) # [N_obj_box_features, in_features_part_det]
        
        fused_obj_box_features = torch.cat((obj_box_features, related_part_features), dim=1)
        fused_part_box_features = torch.cat((part_box_features, related_obj_features), dim=1)

        return fused_obj_box_features, fused_part_box_features
    
    def get_related_obj_box_features(self, part_box, obj_boxes, obj_box_features):
        """
        Find all `obj_boxes` where intersection_area(part_box, obj_box)/area(part_box) >= fusion_thresh
        Arguments:
            part_box (Tensor[4]): bounding box coordinates of the part
            obj_boxes (Tensor[N, 4]): bounding box coordinates of N obj_boxes from which some are selected
            obj_box_features (Tensor[N, in_features_obj_det]): corresponding box_features of the N obj boxes
        Returns:
            related_obj_box_features (Tensor[N', in_features_obj_det]): box_features of obj_boxes that satisfy the condition
                Note: if no related obj_box is found, return zero-tensor of shape [1, in_features_obj_det]
        """
        # TODO: This part is very very slow. Need to optimize it

        N = obj_box_features.shape[0]
        # area_thresh = get_area(part_box)*self.fusion_thresh
        # selected_idx = [i for i in range(N) if get_intersection_area(part_box, obj_boxes[i]) >= area_thresh]
        selected_idx = [i for i in range(N) if is_box_inside(part_box, obj_boxes[i])]
        if selected_idx == []:
            return torch.zeros_like(obj_box_features[0:1])
        return obj_box_features[selected_idx, :]
    
    def get_related_part_box_features(self, obj_box, part_boxes, part_box_features):
        """
        Find all `part_boxes` where intersection_area(obj_box, part_box)/area(part_box) >= fusion_thresh
        Arguments:
            obj_box (Tensor[4]): bounding box coordinates of the obj
            part_boxes (Tensor[N, 4]): bounding box coordinates of N part_boxes from which some are selected
            part_box_features (Tensor[N, in_features_part_det]): corresponding box_features of the N part boxes
        Returns:
            related_part_box_features (Tensor[N', in_features_part_det]): box_features of part_boxes that satisfy the condition
                Note: if no related part_box is found, return zero-tensor of shape [1, in_features_part_det]
        """
        # TODO: This part is very very slow. Need to optimize it
        
        N = part_box_features.shape[0]
        # selected_idx = [i for i in range(N) if get_intersection_area(obj_box, part_boxes[i]) >= get_area(part_boxes[i])*self.fusion_thresh]
        selected_idx = [i for i in range(N) if is_box_inside(part_boxes[i], obj_box)]
        if selected_idx == []:
            return torch.zeros_like(part_box_features[0:1])
        return part_box_features[selected_idx, :]
    
    def get_detections_losses(self, model, class_logits, box_regression, labels, regression_targets, proposals, image_sizes, original_image_sizes, name=''):
        """
        Arguments:
            model (torchvision FasterRCNN model)
            class_logits, box_regression: output of model.roi_heads.box_predictor(box_features)
            labels (List[Tensor[N']]): ground truth class labels for training
            regression_targets (List[Tensor[N', 4]]): ground truth targets for box regression for training
            proposals (List[Tensor[N, 4]]): box proposals
            image_sizes (list[tuple[int, int]]): original shapes of all images
            original_image_sizes (List[Tuple[int, int]]): to postprocess detections
            name (str): append this name to keys in losses
        Returns:
            detections (List[Dict[str, torch.Tensor]]): predicted boxes, labels, and scores (in eval mode)
            detector_losses (Dict): classifier and box_regression losses for final detection (in train mode)
        """
        detections = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        detector_losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detector_losses = {'loss_classifier' + name: loss_classifier, 'loss_box_reg' + name: loss_box_reg}
        else:
            boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_sizes)
            num_images = len(boxes)
            for i in range(num_images):
                detections.append({'boxes': boxes[i], 'labels': labels[i], 'scores': scores[i]})
        
        detections = self.transform.postprocess(detections, image_sizes, original_image_sizes)

        return detections, detector_losses
