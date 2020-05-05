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
        
        fusion_thresh: threshold at which we consider a part/object to be related to another object/part (default: 0.9)
    """
    def __init__(self, obj_n_classes, part_n_classes, fusion_thresh=0.9):
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
            
            related_part_box_feats, related_obj_box_feats = self.get_related_box_features(img_obj_proposals, img_part_proposals,
                                                                                          img_obj_box_features, img_part_box_features)
            related_obj_features.append(related_obj_box_feats)
            related_part_features.append(related_part_box_feats)

        related_obj_features = torch.cat(related_obj_features, dim=0) # [N_part_box_features, in_features_obj_det]
        related_part_features = torch.cat(related_part_features, dim=0) # [N_obj_box_features, in_features_part_det]
        
        fused_obj_box_features = torch.cat((obj_box_features, related_part_features), dim=1)
        fused_part_box_features = torch.cat((part_box_features, related_obj_features), dim=1)

        return fused_obj_box_features, fused_part_box_features
    
    def get_related_box_features_slow(self, obj_boxes, part_boxes, obj_box_features, part_box_features, reduction='mean'):
        """
        see get_related_box_features() for efficient implementation
        For any part_box related `obj_boxes` are those where intersection_area(part_box, obj_box)/area(part_box) >= fusion_thresh
        For any obj_box related `part_boxes` are those where intersection_area(part_box, obj_box)/area(part_box) >= fusion_thresh
            NOTE: Currently using fusion_thresh=1.0 for fast computations
        Arguments:
            obj_boxes (Tensor[N1, 4]): bounding box coordinates of obj_boxes
            part_boxes (Tensor[N2, 4]): bounding box coordinates of part_boxes
            obj_box_features (Tensor[N1, in_features_obj_det]): corresponding box_features of the N1 obj boxes
            part_box_features (Tensor[N2, in_features_part_det]): corresponding box_features of the N2 part boxes
            reduction (string): method to combine features of all related objects/parts for a part/object (default: mean)
                TODO: Support for learning score using attention and doing weighted average
        Returns:
            related_part_box_features (Tensor[N1, in_features_part_det]): mean of box_features of part_boxes related to the N1 obj boxes
            related_obj_box_features (Tensor[N2, in_features_part_det]): mean of box_features of obj_boxes related to the N2 part boxes
                Note: if no related {obj/part}_box is found, return zero-tensor is considered
        """
        N1, N2 = obj_box_features.shape[0], part_box_features.shape[0]
        related_part_idxs = [[] for _ in range(N1)] # list of idxs of related parts for ith obj
        related_obj_idxs = [[] for _ in range(N2)] # list of idxs of related objects for ith part
        for p_idx in range(N2):
            # area_thresh = get_area(part_boxes[p_idx])*self.fusion_thresh
            for o_idx in range(N1):
                # if get_intersection_area(part_boxes[p_idx], obj_boxes[o_idx]) >= area_thresh:
                if is_box_inside(part_boxes[p_idx], obj_boxes[o_idx]):
                    related_part_idxs[o_idx].append(p_idx)
                    related_obj_idxs[p_idx].append(o_idx)

        zero_part_feat = torch.zeros_like(part_box_features[0]) # use this if no related parts for an object
        related_part_box_feats = [] # mean of features of related parts for ith object
        for o_idx in range(N1):
            if related_part_idxs[o_idx] == []:
                related_part_box_feats.append(zero_part_feat)
            else:
                related_part_box_feats.append(part_box_features[related_part_idxs[o_idx]].mean(0))
        related_part_box_feats = torch.stack(related_part_box_feats, dim=0) # [N1, in_features_part_det]

        zero_obj_feat = torch.zeros_like(obj_box_features[0]) # use this if no related objects for a part
        related_obj_box_feats = [] # mean of features of related objects for ith part
        for p_idx in range(N2):
            if related_obj_idxs[p_idx] == []:
                related_obj_box_feats.append(zero_obj_feat)
            else:
                related_obj_box_feats.append(obj_box_features[related_obj_idxs[p_idx]].mean(0))
        related_obj_box_feats = torch.stack(related_obj_box_feats, dim=0)

        return related_part_box_feats, related_obj_box_feats

    def get_related_box_features(self, obj_boxes, part_boxes, obj_box_features, part_box_features, reduction='mean'):
        """
        For any part_box related `obj_boxes` are those where intersection_area(part_box, obj_box)/area(part_box) >= fusion_thresh
        For any obj_box related `part_boxes` are those where intersection_area(part_box, obj_box)/area(part_box) >= fusion_thresh
        Arguments:
            obj_boxes (Tensor[N1, 4]): bounding box coordinates of obj_boxes
            part_boxes (Tensor[N2, 4]): bounding box coordinates of part_boxes
            obj_box_features (Tensor[N1, in_features_obj_det]): corresponding box_features of the N1 obj boxes
            part_box_features (Tensor[N2, in_features_part_det]): corresponding box_features of the N2 part boxes
            reduction (string): method to combine features of all related objects/parts for a part/object (default: mean)
                TODO: Support for learning score using attention and doing weighted average
                TODO: Generify to consider other reduction techniques (may have to implement using loops, not directly using vectors)
        Returns:
            related_part_box_features (Tensor[N1, in_features_part_det]): mean of box_features of part_boxes related to the N1 obj boxes
            related_obj_box_features (Tensor[N2, in_features_part_det]): mean of box_features of obj_boxes related to the N2 part boxes
                Note: if no related {obj/part}_box is found, return zero-tensor is considered
        """
        N1, N2 = obj_boxes.shape[0], part_boxes.shape[0]
        part_box_areas = (part_boxes[:, 3] - part_boxes[:, 1]) * (part_boxes[:, 2] - part_boxes[:, 0]) # Tensor[N2]

        # top-left point of interection area corresponding to each part-object pair
        temp = torch.stack([obj_boxes[:, :2].contiguous().view(-1).repeat(N2), part_boxes[:, :2].repeat(1, N1).view(-1)], dim=1)
        intersect_top_left = temp.max(dim=1)[0].view(N2, N1, 2)
        
        # bottom-right point of interection area corresponding to each part-object pair
        temp = torch.stack([obj_boxes[:, 2:].contiguous().view(-1).repeat(N2), part_boxes[:, 2:].repeat(1, N1).view(-1)], dim=1)
        intersect_bottom_right = temp.min(dim=1)[0].view(N2, N1, 2)

        overlap_sides = torch.clamp(intersect_bottom_right - intersect_top_left, min=0.0) # make negative side lengths zero
        overlap_areas = overlap_sides[:, :, 0] * overlap_sides[:, :, 1] # calculate overlap area for each part-object pair (Tensor[N2, N1])
        
        # part_overlap_fraction's [i,j]th element is fraction of part[i]'s area overlapped with obj[j] [Assumption: each part area > 0.0]
        part_overlap_fraction = overlap_areas / part_box_areas.view(N2, 1) # Tensor[N2, N1]
        related_part_obj_pairs = (part_overlap_fraction >= self.fusion_thresh).float()

        part_wise_obj_overlaps = related_part_obj_pairs.sum(dim=1) # Tensor[N2]
        part_wise_obj_overlaps[part_wise_obj_overlaps == 0.0] = 1.0 # to avoid division by 0 while taking mean for parts not overlapping with any obj
        related_obj_box_feats = torch.matmul(related_part_obj_pairs, obj_box_features) / part_wise_obj_overlaps.view(N2, 1)
        
        obj_wise_part_overlaps = related_part_obj_pairs.sum(dim=0) # Tensor[N1]
        obj_wise_part_overlaps[obj_wise_part_overlaps == 0.0] = 1.0 # to avoid division by 0 while taking mean for obj not overlapping with any parts
        related_part_box_feats = torch.matmul(related_part_obj_pairs.t(), part_box_features) / obj_wise_part_overlaps.view(N1, 1)

        return related_part_box_feats, related_obj_box_feats
        
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
