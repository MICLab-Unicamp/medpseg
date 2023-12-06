import torch
import torch.nn as nn


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(
        a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(
        a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) *
                         (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    # , **kwargs):
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        # assuming all image sizes are the same, which it is
        anchor = anchors[0, :, :]
        dtype = anchors.dtype
        device = classifications.device

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j].to(device)
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:

                alpha_factor = torch.ones_like(classification) * alpha
                alpha_factor = alpha_factor.to(device)
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * \
                    torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                regression_losses.append(torch.tensor(0).to(dtype).to(device))
                classification_losses.append(cls_loss.sum())

                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            targets = targets.to(device)

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices,
                    assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            alpha_factor = alpha_factor.to(device)

            alpha_factor = torch.where(
                torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(
                torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) +
                    (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            zeros = zeros.to(device)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(
                cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                print('positives')
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:,
                                                 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:,
                                                  3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack(
                    (targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(
                    targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(
                    torch.tensor(0).to(dtype).to(device))

        # debug
        # imgs = kwargs.get('imgs', None)
        # if imgs is not None:
        #     regressBoxes = BBoxTransform()
        #     clipBoxes = ClipBoxes()
        #     obj_list = kwargs.get('obj_list', None)
        #     out = postprocess(imgs.detach(),
        #                       torch.stack([anchors[0]] * imgs.shape[0], 0).detach(
        #     ), regressions.detach(), classifications.detach(),
        #         regressBoxes, clipBoxes,
        #         0.5, 0.3)
        #     imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
        #     imgs = ((imgs * [0.229, 0.224, 0.225] +
        #              [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
        #     imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
        #     display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses).mean(dim=0, keepdim=True)
