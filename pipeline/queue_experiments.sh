# different lr
# dvc exp run --queue -S "model.learning_rate=5e-3"  -n "esm2_lora_base_lr5e-3"
# dvc exp run --queue -S "model.learning_rate=5e-4"  -n "esm2_lora_base_lr5e-4"
# dvc exp run --queue -S "model.learning_rate=5e-5"  -n "esm2_lora_base_lr5e-5"
# # different rank and alpha
# dvc exp run --queue -S "model.lora_rank=4" -S "model.lora_alpha=8" -n "esm2_lora_base_rank4_alpha8"
# dvc exp run --queue -S "model.lora_rank=16" -S "model.lora_alpha=32" -n "esm2_lora_base_rank4_alpha16"
# dvc exp run --queue -S "model.lora_rank=2" -S "model.lora_alpha=4" -n "esm2_lora_base_rank4_alpha32"
# # different head numbers
# dvc exp run --queue -S "model.contact_head_dim=2" -n "esm2_lora_base_head2"
# dvc exp run --queue -S "model.contact_head_dim=8" -n "esm2_lora_base_head8"
# dvc exp run --queue -S "model.contact_head_dim=16" -n "esm2_lora_base_head16"
# focal loss
# dvc exp run --queue -S "model=esm2_lora_contact_600m_focal" -n "lora_focal_a0.9_g2"
# dvc exp run --queue -S "model=esm2_lora_contact_600m_focal"  -S "model.loss.focal_alpha=0.75" -n "lora_focal_loss_a0.75_g2"
# dvc exp run --queue -S "model=esm2_lora_contact_600m_focal" -S "model.loss.focal_alpha=-1" -S "model.loss.focal_gamma=3" -n "lora_focal_loss_a-1_g3"
# dvc exp run --queue -S "model=esm2_lora_contact_600m_focal"  -S "model.loss.focal_alpha=-1" -S "model.loss.focal_gamma=2" -n "lora_focal_loss_a-1_g2"
# dvc exp run --queue -S "model=esm2_lora_contact_600m_focal"  -S "model.loss.focal_alpha=-1" -S "model.loss.focal_gamma=1.5" -n "lora_focal_loss_a-1_g1.5"
# dvc exp run --queue -n "paired_base_lr5e4_lora_freezed"
# dvc exp run --queue -S "model.unfreeze_lora=true" -n "paired_base_lr5e4_lora_unfreezed"
# dvc exp run --queue -S "model=esm2_lora_template_improvements" -n "paired_base_lr5e4_improv"
# dvc exp run --queue -S "model=esm2_lora_template_improvements" -S "model.use_weighted_attn_mask=false" -n "paired_base_lr5e4_improv_weighted_attn_mask"
# dvc exp run --queue -S "model=esm2_lora_template_improvements" -S "model.unfreeze_lora=true" -n "paired_improv_lora_unfreezed"
# dvc exp run --queue -S "model=esm2_lora_template_improvements" -S "model.unfreeze_lora=true" -S "model.contacthead_init_from=''" -n "paired__improv_lora_unfreezed_wo_ch_init"
# dvc exp run --queue -S "model=esm2_lora_template_improvements" -S "model.unfreeze_lora=true" -S "model.contacthead_init_from=''" -n "paired__improv_lora_unfreezed_wo_ch_init_lr_multiply"
# dvc exp run --queue -S "model=esm2_lora_template_improvements" -S "model.unfreeze_lora=true" -S "model.contacthead_init_from=''" -S "model.gated_cross_attn_init=-1" -S "model.gate_lr_multiplier=300" -S "model.gated_cross_attn_init=-1.5" -S "model.weighted_attn_mask_alpha_init=-2" -S "model.alpha_lr_multiplier=300" -n "increase_gates"
# dvc exp run --queue -S "model=esm2_lora_template_improvements" -S "model.unfreeze_lora=true" -S "model.contacthead_init_from=''" -S "model.gated_cross_attn_init=-1" -S "model.gate_lr_multiplier=300" -S "model.gated_cross_attn_init=-1.5" -S "model.weighted_attn_mask_alpha_init=-2" -S "model.alpha_lr_multiplier=300" -S "model.backbone_init_from=''" -n "increase_gates_wo_backbone_init"
dvc exp run --queue -S "model=esm2_lora_template_improvements" -S "model.unfreeze_lora=false"  -S "model.use_gated_cross_attn=false"  -S "model.weighted_attn_mask_alpha_init=-2" -S "model.alpha_lr_multiplier=100" -S "model.contacthead_init_from=''" -n "increase_alpha_wo_gate_wo_ch_init"



dvc queue status