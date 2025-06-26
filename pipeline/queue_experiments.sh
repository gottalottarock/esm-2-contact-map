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
dvc exp run --queue -n "paired_base_lr5e4_lora_freezed"
dvc exp run --queue -S "model.unfreeze_lora=true" -n "paired_base_lr5e4_lora_unfreezed"
dvc queue status