class ComplexClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=256, dropout=0.3, use_bn=True, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

        if use_residual and in_dim != hidden_dim:
            self.res_fc = nn.Linear(in_dim, hidden_dim)
        else:
            self.res_fc = nn.Identity()

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)
        h2 = self.fc2(h)
        h2 = self.bn2(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)
        if self.use_residual:
            h2 = h2 + self.res_fc(x)
        out = self.fc_out(h2)
        return out

class TFBVS(nn.Module):
    def __init__(
            self,
            backbone,
            transformer,
            num_queries,
            aux_loss=False,
            iter_update=False,
            query_dim=2,
            num_feature_levels=1,
            nheads=8,
            two_stage_type="no", 
            dec_pred_bbox_embed_share=True,
            two_stage_class_embed_share=True,
            two_stage_bbox_embed_share=True,
            num_patterns=0,
            dn_number=100,
            dn_box_noise_scale=0.4,
            dn_label_noise_ratio=0.5,
            dn_labelbook_size=100,
            text_encoder_type="bert-base-uncased",
            sub_sentence_present=True,
            max_text_len=256,
            num_classes=1,
            dropout=0.0,
            dec_lora=False,
            enc_lora=False,
            lora_rank=16,
            pixel_decoder=None,
            mask_decoder=None,
            temporal_layer=3,
            tracking_alpha=0.1,
            full_tune=False,
            trainable_key_list=[],
            use_ot_prompt=False,
            textfree_infer=False,
            t_dim=None,
            num_prompt_tokens=8,
            ot_hidden=512,
            ot_pool='mean',       
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present
        self.class_proj = ComplexClassifier(self.hidden_dim, num_classes=4, hidden_dim=256, dropout=0.3)
        self.query_dim = query_dim
        assert query_dim == 4
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)
        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = None
        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        _class_embed = ContrastiveEmbed()
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for _ in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for _ in range(transformer.num_decoder_layers)]
        class_embed_layerlist = [_class_embed for _ in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"]
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self.ref_point_head = self.transformer.decoder.ref_point_head
        self._reset_parameters()
        if enc_lora or dec_lora:
            if enc_lora:
                _set_lora_linear(self.transformer.encoder, lora_rank)
            if dec_lora:
                _set_lora_linear(self.transformer.decoder, lora_rank)

        if pixel_decoder and mask_decoder:
            self.mask = True
            self.pixel_decoder = pixel_decoder
            self.mask_decoder = mask_decoder
        else:
            self.mask = False

        _temporal_decoder = TemporalDecoder(hidden_dim, nheads, temporal_layer, dropout)
        temporal_decoder_list = [_temporal_decoder for _ in range(transformer.num_decoder_layers)]
        self.temporal_decoder = nn.ModuleList(temporal_decoder_list)
        self.tracking_alpha = tracking_alpha
        self.register_buffer("mu_track",  torch.zeros(1, num_queries))
        self.register_buffer("var_track", torch.zeros(1, num_queries))
        self.alpha_min  = 0.05
        self.alpha_max  = tracking_alpha
        self.beta_sigma = 5.0
        self.sigma_0    = 0.05
        self.gamma_gate = nn.Parameter(torch.tensor(0.30))
        self.lmbd_ema   = 0.25
        if not full_tune:
            for n, p in self.named_parameters():
                if any(keyword in n for keyword in trainable_key_list):
                    continue
                p.requires_grad_(False)

        self.use_ot_prompt  = use_ot_prompt
        self.textfree_infer = textfree_infer
        self.t_dim = t_dim if t_dim is not None else self.hidden_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.ot_map = OTMap(in_dim=self.hidden_dim, t_dim=self.t_dim, hidden=ot_hidden, pool=ot_pool)
        self.prompt_projector = PromptProjector(self.t_dim, self.hidden_dim, num_tokens=self.num_prompt_tokens)

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    def _build_video_frame_tokens(self, srcs, mask_bt_hw):
        feat = srcs[-1]                               # [(B*T), D, H, W]
        B_T, C, H, W = feat.shape
        B, T = mask_bt_hw.shape[:2]
        feat = feat.view(B, T, C, H, W)
        feat_sp = feat.mean(dim=(-2, -1))            # [B, T, D]
        valid_bt = (~mask_bt_hw).any(dim=(-1, -2))   # [B, T]
        feat_sp = feat_sp * valid_bt[..., None].float()
        return feat_sp                                # [B, T, D]
    def _map_seq_no_pool(self, v_seq_bt, video_labels=None):
        if torch.isnan(v_seq_bt).any() or torch.isinf(v_seq_bt).any():
            v_seq_bt = torch.nan_to_num(v_seq_bt, nan=0.0, posinf=1e6, neginf=-1e6)

        B, T, D = v_seq_bt.shape
        v_flat = v_seq_bt.reshape(B * T, D)
        denom = v_flat.abs().max(dim=1, keepdim=True)[0] + 1e-8
        v_norm = v_flat / denom  # [B*T, D]
        if video_labels is not None:
          
            labels_expanded = video_labels.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]
            label_emb = self.ot_map.label_embed(labels_expanded)  # [B*T, D]
            v_cat = torch.cat([v_norm, label_emb], dim=-1)        # [B*T, 2D]
            hat_T_seq_flat = self.ot_map.mlp(v_cat)               # [B*T, t_dim]
        else:
            hat_T_seq_flat = self.ot_map.mlp(v_norm)

        hat_T_seq = hat_T_seq_flat.view(B, T, -1)                 # [B, T, t_dim]
        return hat_T_seq

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: NestedTensor, captions, targets, **kw):
        device = samples.device
        if samples.mask.dim() == 4:
            B, T, H, W = samples.mask.shape
        elif samples.mask.dim() == 3:
            B, H, W = samples.mask.shape
            T = 1
            samples.mask = samples.mask.unsqueeze(1)  # [B, 1, H, W]
            samples.tensors = samples.tensors.unsqueeze(1)  # [B, 1, 3, H, W]
        else:
            raise ValueError("samples.mask shape error: {}".format(samples.mask.shape))
        samples.tensors = rearrange(samples.tensors, 'b t c h w -> (b t) c h w')
        samples.mask    = rearrange(samples.mask,    'b t h w -> (b t) h w')
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        poss = poss[-3:]
        srcs, masks = [], []
        for l, feat in enumerate(features[-3:]):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src); masks.append(mask); poss.append(pos_l)
        def _is_empty_caption_list(caps):
            if caps is None: return True
            if isinstance(caps, list):
                return all((c is None) or (str(c).strip() == "") for c in caps)
            return False
        no_text_input = _is_empty_caption_list(captions)
        use_pseudo = (self.use_ot_prompt and no_text_input) or ((not self.training) and self.textfree_infer and no_text_input)
        mask_bt_hw = rearrange(samples.mask, '(b t) h w -> b t h w', b=B)    # [B,T,H,W]
        v_seq_bt = self._build_video_frame_tokens(srcs, mask_bt_hw)          # [B,T,D]
        if isinstance(targets, dict) and "classes" in targets:
            classes = targets["classes"].to(samples.device)  # [B]
        elif isinstance(targets, list):
            classes = torch.stack([t['classes'][0] for t in targets]).to(samples.device)  # [B]
        else:
            raise RuntimeError("targets NO classes")
        video_feat = v_seq_bt.mean(dim=1) 
        video_logits = self.class_proj(video_feat)  # [B, num_classes]
        #print(f"video_logits shape: {video_logits.shape}")
        if isinstance(targets, dict) and "classes" in targets:
            classes = targets["classes"].to(samples.device)
        elif isinstance(targets, list):
            classes = torch.stack([t['classes'][0] for t in targets]).to(samples.device)
        hat_T_global = self.ot_map(v_seq_bt, labels=classes) 
        hat_T_seq = self._map_seq_no_pool(v_seq_bt, video_labels=classes) 
        if use_pseudo:
            prompt_tokens = self.prompt_projector(hat_T_global)               # [B, M, d_model]
            encoded_text_pseudo = repeat(prompt_tokens, 'b m d -> (b t) m d', t=T)
            text_token_mask_pseudo = torch.ones(encoded_text_pseudo.size()[:2], dtype=torch.bool, device=device)
            attn_mask_pseudo = torch.ones(
                encoded_text_pseudo.size(0), encoded_text_pseudo.size(1), encoded_text_pseudo.size(1),
                dtype=torch.bool, device=device
            )
            text_dict_used = {
                "encoded_text": encoded_text_pseudo,                # [(B*T), M, d]
                "text_token_mask": text_token_mask_pseudo,          # [(B*T), M]
                "position_ids": None,
                "text_self_attention_masks": attn_mask_pseudo,      # [(B*T), M, M]
                "embed_text": prompt_tokens,                        # [B, M, d]
            }
            text_cls = None
        else:
            captions = [t if (t and t.endswith(".")) else (str(t) + ".") for t in captions]
            tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
            text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )
            if text_self_attention_masks.shape[1] > self.max_text_len:
                text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"]      = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

            if self.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"]   = position_ids
            else:
                tokenized_for_encoder = tokenized

            bert_output     = self.bert(**tokenized_for_encoder)                # [B, L, 768]
            encoded_text    = self.feat_map(bert_output["last_hidden_state"])   # [B, L, d_model]
            text_token_mask = tokenized.attention_mask.bool()                   # [B, L]
            if encoded_text.shape[1] > self.max_text_len:
                encoded_text    = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids    = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]

            text_dict_used = {
                "encoded_text": repeat(encoded_text, 'b l d -> (b t) l d', t=T),
                "text_token_mask": repeat(text_token_mask, 'b l -> (b t) l', t=T),
                "position_ids": repeat(position_ids, 'b l -> (b t) l', t=T),
                "text_self_attention_masks": repeat(text_self_attention_masks, 'b l1 l2 -> (b t) l1 l2', t=T),
                "embed_text": encoded_text,                                      # [B, L, d]
            }
            text_cls = encoded_text[:, 0, :]  # [B, d_model]
        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal, memory_enc, selected_query_list = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict_used
        )
        outputs_coord_list, outputs_mask_list, outputs_class_list = [], [], []
        if self.mask:
            memory_enc.insert(0, features[0].tensors)
            mask_features = self.pixel_decoder(memory_enc[-1], memory_enc[:-1][::-1])
            mask_features_flat = mask_features.flatten(2).permute(2, 0, 1)
            mask_features_padding = features[0].mask.flatten(1)
            start_index   = torch.tensor(0, device=device)
            spatial_shapes = torch.as_tensor(mask_features.shape[-2:], device=device)[None, :]
            valid_ratios   = self.transformer.get_valid_ratio(features[0].mask)
            valid_ratios   = torch.cat([valid_ratios, valid_ratios], -1)[:, None, :]
        if T > 1:
            text_feat = rearrange(text_dict_used['encoded_text'], '(b t) l d -> t b l d', t=T)
            sent_feat = text_feat[:, :, 0]
            hs_tracked, ref_tracked = [], []
            alpha = self.tracking_alpha
            for l in range(len(hs)):
                layer_hs = rearrange(hs[l], '(b t) q d -> t b q d', t=T)
                layer_ref = rearrange(reference[l], '(b t) q d -> t b q d', t=T)
                layer_hs_tracked = [layer_hs[0]]
                layer_ref_tracked = [layer_ref[0]]
                hs_memory = layer_hs[0]
                for t in range(1, T):
                    ind = self.match_from_embds(hs_memory, layer_hs[t])
                    ind = torch.tensor(ind, device=device)
                    hs_sorted = torch.gather(layer_hs[t], 1, ind[:, :, None].expand(-1, -1, self.hidden_dim))
                    ref_sorted = torch.gather(layer_ref[t], 1, ind[:, :, None].expand(-1, -1, 4))
                    conf = torch.matmul(hs_sorted, text_feat[t].transpose(1, 2)).max(-1)[0]   # (B,Q)
                    conf = (conf - conf.min(1, keepdim=True)[0]) / \
                           (conf.max(1, keepdim=True)[0] - conf.min(1, keepdim=True)[0] + 1e-6)
                    sim = F.cosine_similarity(hs_memory, hs_sorted, dim=-1)                  # (B,Q)

                    Q_now = sim.size(1)
                    if self.mu_track.size(1) != Q_now:
                        self.mu_track  = torch.zeros(1, Q_now, device=sim.device)
                        self.var_track = torch.zeros(1, Q_now, device=sim.device)

                    self.mu_track  = (1 - self.lmbd_ema) * self.mu_track + self.lmbd_ema * sim
                    self.var_track = (1 - self.lmbd_ema) * self.var_track + self.lmbd_ema * (sim - self.mu_track).pow(2)
                    sigma = (self.var_track + 1e-6).sqrt()                                   # (1,Q)

                    alpha_sigma = self.alpha_min + (self.alpha_max - self.alpha_min) * \
                                  torch.sigmoid(self.beta_sigma * (sigma - self.sigma_0))
                    conf_gate = self.gamma_gate + (1 - self.gamma_gate) * conf               # (B,Q)
                    alpha_eff = (alpha_sigma * conf_gate).unsqueeze(-1)                      # (B,Q,1)

                    hs_memory = (1 - alpha_eff) * hs_memory + alpha_eff * hs_sorted
                    layer_hs_tracked.append(hs_sorted)
                    layer_ref_tracked.append(ref_sorted)

                layer_hs_tracked  = torch.stack(layer_hs_tracked, 0)
                layer_ref_tracked = torch.stack(layer_ref_tracked, 0)
                hs_tracked.append(rearrange(layer_hs_tracked,  't b q d -> (b t) q d'))
                ref_tracked.append(rearrange(layer_ref_tracked, 't b q d -> (b t) q d'))
            hs = hs_tracked
            reference[:-1] = ref_tracked
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_cls_embed, layer_temp, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, self.class_embed, self.temporal_decoder, hs)
        ):
            if T > 1:
                text_feat = rearrange(text_dict_used['encoded_text'], '(b t) l d -> t b l d', t=T)
                sent_feat = text_feat[:, :, 0]
                layer_sent_feat = repeat(sent_feat, 't b d -> t (b q) d', q=layer_hs.shape[1])
                layer_hs = rearrange(layer_hs, '(b t) q d -> t (b q) d', b=B)
                layer_hs = layer_temp(layer_hs, layer_sent_feat)  # t, b*q, d
                layer_hs = rearrange(layer_hs, 't (b q) d -> (b t) q d', b=B)
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_coord_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_coord = layer_coord_unsig.sigmoid()
            outputs_coord_list.append(layer_coord)
            if self.mask:
                query_ref_input = layer_ref_sig * valid_ratios
                query_sine_embed = gen_sineembed_for_position(query_ref_input)
                query_pos  = self.ref_point_head(query_sine_embed).transpose(0, 1)  # nq, bs, 256
                query_ref_input = query_ref_input.transpose(0, 1)[:, :, None, :]
                mask_embed = layer_hs.transpose(0, 1)
                for mask_layer in self.mask_decoder:
                    mask_embed = mask_layer(
                        tgt=mask_embed,
                        tgt_query_pos=query_pos,
                        tgt_reference_points=query_ref_input,
                        memory_text=text_dict_used["encoded_text"],
                        text_attention_mask=~text_dict_used["text_token_mask"],
                        memory=mask_features_flat,
                        memory_key_padding_mask=mask_features_padding,
                        memory_level_start_index=start_index,
                        memory_spatial_shapes=spatial_shapes,
                    )
                mask_embed = mask_embed.transpose(0, 1)
                layer_outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_mask_list.append(layer_outputs_mask)
            layer_class = torch.max(layer_cls_embed(layer_hs, text_dict_used), dim=-1, keepdim=True)[0] 
            outputs_class_list.append(layer_class)
        for l in range(len(hs)):
            outputs_class_list[l] = rearrange(outputs_class_list[l], '(b t) q k -> b t q k', b=B)
            outputs_coord_list[l] = rearrange(outputs_coord_list[l], '(b t) q c -> b t q c', b=B)
        out = {"pred_logits": outputs_class_list[-1], "pred_boxes": outputs_coord_list[-1]}
        if self.mask:
            for l in range(len(hs)):
                outputs_mask_list[l] = rearrange(outputs_mask_list[l], '(b t) q h w -> b t q h w', b=B)
            out.update({"pred_masks": outputs_mask_list[-1]})

        if hs_enc is not None:
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict_used).max(-1, keepdim=True)[0]
            interm_class = rearrange(interm_class, '(b t) q k -> b t q k', b=B)
            interm_coord = rearrange(ref_enc[-1], '(b t) q k -> b t q k', b=B)
            interm_proposal = rearrange(init_box_proposal, '(b t) q c -> b t q c', b=B)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': interm_proposal}
            if self.mask:
                query_ref_input = init_box_proposal * valid_ratios
                query_sine_embed = gen_sineembed_for_position(query_ref_input)
                query_pos  = self.ref_point_head(query_sine_embed).transpose(0, 1)
                query_ref_input = query_ref_input.transpose(0, 1)[:, :, None, :]
                mask_embed = hs_enc[-1].transpose(0, 1)
                for mask_layer in self.mask_decoder:
                    mask_embed = mask_layer(
                        tgt=mask_embed,
                        tgt_query_pos=query_pos,
                        tgt_reference_points=query_ref_input,
                        memory_text=text_dict_used["encoded_text"],
                        text_attention_mask=~text_dict_used["text_token_mask"],
                        memory=mask_features_flat,
                        memory_key_padding_mask=mask_features_padding,
                        memory_level_start_index=start_index,
                        memory_spatial_shapes=spatial_shapes,
                    )
                mask_embed = mask_embed.transpose(0, 1)
                interm_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                interm_mask = rearrange(interm_mask, '(b t) q h w -> b t q h w', b=B)
                out['interm_outputs'].update({"pred_masks": interm_mask})
                out['interm_outputs_for_matching_pre'].update({"pred_masks": interm_mask})
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class_list, outputs_coord_list, outputs_mask_list)
        out['hat_T_global'] = hat_T_global                
        out['hat_T_seq']    = hat_T_seq                 
        out['text_cls']     = text_cls                
        out["video_logits"] = video_logits 
        
        return out

    def infer(self, all_samples: NestedTensor, captions, targets, crop_len=36, **kw):
        all_samples = copy.deepcopy(all_samples)
        if isinstance(all_samples, (list, torch.Tensor)):
            all_samples = nested_tensor_from_tensor_list(all_samples)

        B, T, H, W = all_samples.mask.shape
        assert B == 1, "Inference only supports for batch size = 1"
        device = all_samples.device
        nl = self.transformer.num_decoder_layers - 1

        def _is_empty_caption_list(caps):
            if caps is None: return True
            if isinstance(caps, list):
                return all((c is None) or (str(c).strip() == "") for c in caps)
            return False

        no_text_input = _is_empty_caption_list(captions)
        use_pseudo = (self.use_ot_prompt and no_text_input) or ((not self.training) and self.textfree_infer and no_text_input)

        if not use_pseudo:
            captions = [""] if captions is None else [t if t.endswith(".") else (str(t) + ".") for t in captions]
            tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
            text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )
            if text_self_attention_masks.shape[1] > self.max_text_len:
                text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"]      = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

            if self.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"]   = position_ids
            else:
                tokenized_for_encoder = tokenized

            bert_output     = self.bert(**tokenized_for_encoder)
            encoded_text0   = self.feat_map(bert_output["last_hidden_state"])
            text_token_mask0 = tokenized.attention_mask.bool()
            if encoded_text0.shape[1] > self.max_text_len:
                encoded_text0   = encoded_text0[:, : self.max_text_len, :]
                text_token_mask0 = text_token_mask0[:, : self.max_text_len]
                position_ids     = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]

        all_features, all_hs, all_reference, all_memory_enc = [], [], [], []
        all_text_used = {"encoded_text": [], "text_token_mask": [], "position_ids": [], "text_self_attention_masks": []}

        for clip_id in range(0, T, crop_len):
            samples = NestedTensor(
                all_samples.tensors[:, clip_id: clip_id + crop_len],
                all_samples.mask[:, clip_id: clip_id + crop_len]
            )
            samples.tensors = rearrange(samples.tensors, 'b t c h w -> (b t) c h w')
            samples.mask    = rearrange(samples.mask,    'b t h w -> (b t) h w')
            t = samples.tensors.shape[0]

            features, poss = self.backbone(samples); poss = poss[-3:]

            srcs, masks = [], []
            for l, feat in enumerate(features[-3:]):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src); masks.append(mask); poss.append(pos_l)

            if use_pseudo:
                frame_tokens = srcs[-1].mean(dim=[2, 3])                    # [(B*t), D]
                v_seq_bt = rearrange(frame_tokens, '(b t) d -> b t d', b=B) # [B, t, D]
                video_feats = v_seq_bt.mean(dim=1)                          # [B, D]
                class_logits = self.class_proj(video_feats)                 
                pred_labels = class_logits.argmax(dim=-1)                   
                hat_T = self.ot_map(v_seq_bt, labels=pred_labels)           # [B, t_dim]
                prompt_tokens = self.prompt_projector(hat_T)                # [B, M, d_model]
                encoded_text_clip = repeat(prompt_tokens, 'b m d -> (b t) m d', t=t)
                text_token_mask_clip = torch.ones(encoded_text_clip.size()[:2], dtype=torch.bool, device=device)
                text_self_attn_clip  = torch.ones(
                    encoded_text_clip.size(0), encoded_text_clip.size(1), encoded_text_clip.size(1),
                    dtype=torch.bool, device=device
                )
                text_dict_used_clip = {
                    "encoded_text": encoded_text_clip,
                    "text_token_mask": text_token_mask_clip,
                    "position_ids": None,
                    "text_self_attention_masks": text_self_attn_clip,
                }
            else:
                encoded_text_clip = repeat(encoded_text0,   'b l d -> (b t) l d', t=t)
                text_token_mask_clip = repeat(text_token_mask0, 'b l -> (b t) l', t=t)
                position_ids_clip = repeat(position_ids, 'b l -> (b t) l', t=t)
                text_self_attn_clip = repeat(text_self_attention_masks, 'b l1 l2 -> (b t) l1 l2', t=t)
                text_dict_used_clip = {
                    "encoded_text": encoded_text_clip,
                    "text_token_mask": text_token_mask_clip,
                    "position_ids": position_ids_clip,
                    "text_self_attention_masks": text_self_attn_clip,
                }
            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            hs, reference, hs_enc, ref_enc, init_box_proposal, memory_enc, selected_query_list = self.transformer(
                srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict_used_clip
            )
            all_hs.append(hs[nl]); all_reference.append(reference[nl])
            all_memory_enc.append(memory_enc); all_features.append(features[0])

            for k in all_text_used.keys():
                all_text_used[k].append(text_dict_used_clip.get(k, None))

        hs = torch.cat(all_hs, dim=0)                          # [(B*T), q, d]
        reference = torch.cat(all_reference, dim=0)            # [(B*T), q, d]
        memory_enc = [torch.cat(mem, dim=0) for mem in list(zip(*all_memory_enc))]
        features = NestedTensor(*[torch.cat(x, dim=0) for x in list(zip(*[feat.decompose() for feat in all_features]))])

        text_dict_used_full = {}
        for k, vlist in all_text_used.items():
            if all(x is None for x in vlist):
                text_dict_used_full[k] = None
            else:
                vlist = [x for x in vlist if x is not None]
                text_dict_used_full[k] = torch.cat(vlist, dim=0)

        text_feat = rearrange(text_dict_used_full['encoded_text'], '(b t) l d -> t b l d', t=T)
        sent_feat = repeat(text_feat[:, :, 0], 't b d -> t (b q) d', q=hs.shape[1])
        alpha = self.tracking_alpha

        hs = rearrange(hs, '(b t) q d -> t b q d', t=T)
        reference = rearrange(reference, '(b t) q d -> t b q d', t=T)
        hs_tracked = [hs[0]]; ref_tracked = [reference[0]]; hs_memory = hs[0]
        for ti in range(1, T):
            ind = self.match_from_embds(hs_memory, hs[ti])
            ind = torch.tensor(ind, device=device)
            hs_sorted  = torch.gather(hs[ti],  1, ind[:, :, None].expand(-1, -1, self.hidden_dim))
            ref_sorted = torch.gather(reference[ti], 1, ind[:, :, None].expand(-1, -1, 4))
            conf = torch.matmul(hs_sorted, text_feat[ti].transpose(1, 2)).max(-1)[0]
            conf = (conf - conf.min(1)[0][:, None]) / (conf.max(1)[0] - conf.min(1)[0] + 1e-6)
            conf = conf.unsqueeze(-1)
            hs_memory = (1 - alpha * conf) * hs_memory + alpha * conf * hs_sorted
            hs_tracked.append(hs_sorted); ref_tracked.append(ref_sorted)

        hs = rearrange(torch.stack(hs_tracked, 0), 't b q d -> t (b q) d', b=B)
        reference = rearrange(torch.stack(ref_tracked, 0), 't b q d -> (b t) q d', b=B)

        hs = self.temporal_decoder[nl](hs, sent_feat)          # [t, b*q, d]
        hs = rearrange(hs, 't (b q) d -> (b t) q d', b=B)
        logits = torch.max(self.class_embed[nl](hs, text_dict_used_full), dim=-1, keepdim=True)[0]
        delta_unsig = self.bbox_embed[nl](hs)
        coord_unsig = delta_unsig + inverse_sigmoid(reference)
        coord = coord_unsig.sigmoid()

        memory_enc.insert(0, features.tensors)
        mask_features = self.pixel_decoder(memory_enc[-1], memory_enc[:-1][::-1])
        mask_features_flat = mask_features.flatten(2).permute(2, 0, 1)
        mask_features_padding = features.mask.flatten(1)
        start_index = torch.tensor(0, device=device)
        spatial_shapes = torch.as_tensor(mask_features.shape[-2:], device=device)[None, :]
        valid_ratios = self.transformer.get_valid_ratio(features.mask)
        valid_ratios = torch.cat([valid_ratios, valid_ratios], -1)[:, None, :]
        query_ref_input = reference * valid_ratios
        query_sine_embed = gen_sineembed_for_position(query_ref_input)
        query_pos = self.ref_point_head(query_sine_embed).transpose(0, 1)
        query_ref_input = query_ref_input.transpose(0, 1)[:, :, None, :]
        mask_embed = hs.transpose(0, 1)

        mask_embed_list = []
        for clip_id in range(0, T, crop_len):
            clip_mask_embed = mask_embed[:, clip_id: clip_id + crop_len]
            for mask_layer in self.mask_decoder:
                clip_mask_embed = mask_layer(
                    tgt=clip_mask_embed,
                    tgt_query_pos=query_pos[:, clip_id: clip_id + crop_len],
                    tgt_reference_points=query_ref_input[:, clip_id: clip_id + crop_len],
                    memory_text=text_dict_used_full["encoded_text"][clip_id: clip_id + crop_len],
                    text_attention_mask=~text_dict_used_full["text_token_mask"][clip_id: clip_id + crop_len],
                    memory=mask_features_flat[:, clip_id: clip_id + crop_len],
                    memory_key_padding_mask=mask_features_padding[clip_id: clip_id + crop_len],
                    memory_level_start_index=start_index,
                    memory_spatial_shapes=spatial_shapes,
                )
            mask_embed_list.append(clip_mask_embed.transpose(0, 1))
        mask_embed = torch.cat(mask_embed_list, dim=0)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        out = {
            "pred_logits": rearrange(logits, '(b t) q k -> b t q k', b=B),
            "pred_boxes":  rearrange(coord,  '(b t) q k -> b t q k', b=B),
            "pred_masks":  rearrange(outputs_mask, '(b t) q h w -> b t q h w', b=B)
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        if self.mask:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

    @torch.no_grad()
    def match_from_embds(self, mem, cur):
        mem = F.normalize(mem, dim=-1)
        cur = F.normalize(cur, dim=-1)
        cos_sim = torch.matmul(mem, cur.transpose(1, 2))
        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        if torch.isnan(C).any() or torch.isinf(C).any():
            print("NaN or Inf detected in C!")
            print(C)
        C = C.cpu()
        indices = np.stack([linear_sum_assignment(c)[1] for c in C], axis=0)
        return indices


def build_TFBVS(args, num_classes):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    hidden_dim = transformer.d_model

    pixel_decoder = FPNSpatialDecoder(hidden_dim, 2 * [hidden_dim] + [backbone.num_channels[0]], args.SegHead.mask_dim)

    mask_dec_layer = DeformableTransformerDecoderLayer(
        hidden_dim,
        n_levels=1,
        n_heads=args.SegHead.n_heads,
        n_points=args.SegHead.n_points,
        dropout=args.SegHead.dropout,
        use_text_cross_attention=args.SegHead.use_text_cross_attention,
    )
    mask_decoder = _get_clones(mask_dec_layer, args.SegHead.n_layers)

    trainable_key_list = ["feat_map", "LayerNorm", "backbone.0.norm0", ".bbox_embed", "lora",
                          "pixel_decoder", "mask_decoder", "classifier"]
    if not args.single_frame:
        trainable_key_list.append("temporal")

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = TFBVS(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        num_classes=num_classes,
        dec_lora=args.dec_lora,
        enc_lora=args.enc_lora,
        lora_rank=args.lora_rank,
        pixel_decoder=pixel_decoder,
        mask_decoder=mask_decoder,
        trainable_key_list=trainable_key_list,
        full_tune=args.full_tune,
        temporal_layer=args.temporal_layer,
        tracking_alpha=args.tracking_alpha,
        dropout=args.dropout,
        use_ot_prompt = getattr(args, 'use_ot_prompt', True),
        textfree_infer = getattr(args, 'textfree_infer', True),
        t_dim = getattr(args, 't_dim', None),
        num_prompt_tokens = getattr(args, 'num_prompt_tokens', 8),
        ot_hidden = getattr(args, 'ot_hidden', 512),
        ot_pool = getattr(args, 'ot_pool', 'mean'),
    )
    checkpoint = torch.load(args.pretrained_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    if not getattr(args, 'use_ot_prompt', False):
        for p in model.ot_map.parameters():
            p.requires_grad_(False)
        for p in model.prompt_projector.parameters():
            p.requires_grad_(False)
    model.use_ot_prompt   = getattr(args, 'use_ot_prompt', False)
    model.textfree_infer  = getattr(args, 'textfree_infer', False)
    model.num_prompt_tokens = getattr(args, 'num_prompt_tokens', 8)
    return model