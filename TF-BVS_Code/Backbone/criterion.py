class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices):
            valid_ind = valid.nonzero().flatten()
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if self.num_classes == 1:
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=2
        ) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  # [B, T, Q, 4]
        # bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)  # [B, Q, T, 4]

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]             # [sum(B), T, 4]
        src_boxes = src_boxes.flatten(0, 1)    # [B*T, 4]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [B*T, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    def loss_video_cls(self, outputs, targets, indices, num_boxes, **kwargs):
        if 'video_logits' not in outputs:
            return {}
        if isinstance(targets, dict) and "classes" in targets:
            video_labels = targets["classes"]
        elif isinstance(targets, list):
            video_labels = torch.stack([t["classes"][0] for t in targets])
        else:
            raise RuntimeError("targets no classes")
        video_logits = outputs["video_logits"]  # [B, num_classes]
        return {'loss_video_cls': F.cross_entropy(video_logits, video_labels)}
    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]      # [B, T, Q, H, W]
        src_masks = src_masks.transpose(1, 2)  # [B, Q, T, H, W]
        src_masks = src_masks[src_idx]         # [B, T, H, W]
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets], size_divisibility=32, split=False
        ).decompose()
        target_masks = target_masks.to(src_masks)
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        nf = target_masks.shape[1]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        valid = valid[:, start::self.mask_out_stride, start::self.mask_out_stride]
        valid = valid[:, None].expand(-1, nf, -1, -1)
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w
        src_y = src_masks.max(dim=2)[0].flatten(1)
        src_x = src_masks.max(dim=3)[0].flatten(1)
        tgt_y = target_masks.max(dim=2)[0].flatten(1)
        tgt_x = target_masks.max(dim=3)[0].flatten(1)
        src_masks = src_masks.flatten(1)       # [b, thw]
        target_masks = target_masks.flatten(1) # [b, thw]
        valid = valid.flatten(1)               # [b, thw]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes, valid=valid),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes, valid=valid),
            "loss_proj": (dice_loss(src_y, tgt_y, num_boxes) + dice_loss(src_x, tgt_x, num_boxes)) / 2
        }
        return losses
    # --------------------- Hungarian indexing helpers ---------------------
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'video_cls': self.loss_video_cls
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    def _loss_ot_regularizer(self, outputs: dict) -> dict:
        losses = {}
        if 'loss_ot_reg' not in self.weight_dict:
            return losses

        hat_T = outputs.get('hat_T_global', None)  # [B, t_dim]
        if hat_T is None:
            return losses

        losses['loss_ot_reg'] = hat_T.pow(2).mean()
        return losses
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1)  # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item()
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'interm_outputs' in outputs:
            indices = self.matcher(outputs['interm_outputs_for_matching_pre'], targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, outputs['interm_outputs'], targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)
        try:
            losses.update(self._loss_ot_regularizer(outputs))
        except Exception:
            pass
        return losses