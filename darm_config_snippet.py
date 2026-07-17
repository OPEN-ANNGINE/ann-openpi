# ============================================================================
# darmR pick-and-place -> pi05 LoRA fine-tune config
# ----------------------------------------------------------------------------
# These snippets go into  openpi/src/openpi/training/config.py.
# They reference names already defined in that file (DataConfig, DataConfigFactory,
# _transforms, ModelTransformFactory, pi0_config, weight_loaders, _optimizer, etc.),
# so paste them in-place rather than importing this file.
# ============================================================================


# ---- 1) Add near the other policy imports at the top of config.py -----------
#        (next to `from openpi.policies import libero_policy`)
#
# from openpi.policies import darm_policy


# ---- 2) Add this class next to LeRobotLiberoDataConfig ----------------------

@dataclasses.dataclass(frozen=True)
class LeRobotDarmDataConfig(DataConfigFactory):
    """Data config for the darmR pick-and-place LeRobot dataset (dual-arm + hands, 3 cams)."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Rename raw dataset columns -> the keys DarmInputs expects.
        # RepackTransform maps {new_key: old_flattened_key}; old keys are the literal
        # LeRobot column names (with dots). The action column is singular ("action").
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/head": "observation.images.head",
                        "observation/wrist_left": "observation.images.wrist_left",
                        "observation/wrist_right": "observation.images.wrist_right",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[darm_policy.DarmInputs(model_type=model_config.model_type)],
            outputs=[darm_policy.DarmOutputs()],
        )

        # darmR actions are ABSOLUTE joint targets (cmd_*), so we do NOT apply a delta
        # transform -- this matches the pi05 DROID joint-position finetune path.
        #
        # If you later want to train on delta joint targets (relative to the current
        # state) while keeping the two hand dims absolute, uncomment the block below.
        # The first 26 action dims align 1:1 with the 26 state dims; the last 2
        # (Right_Hand, Left_Hand) have no state counterpart and stay absolute.
        #
        # delta_action_mask = _transforms.make_bool_mask(26, -2)  # 26 delta, 2 absolute
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# ---- 3) Add this entry inside the `_CONFIGS = [ ... ]` list ------------------

TrainConfig(
    name="pi05_darm_pnp_lora",
    # pi05 flow-matching model, LoRA on both the VLM backbone and the action expert.
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,      # pi05 native action dim; 26-state/28-action pad up to this
        action_horizon=16,  # ~0.53 s @ 30 fps. Raise toward 25-32 for longer open-loop chunks.
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotDarmDataConfig(
        repo_id="darmR_pnp_both",  # dataset dir name under $HF_LEROBOT_HOME (see README)
        base_config=DataConfig(
            prompt_from_task=True,          # single task string -> prompt
            action_sequence_keys=("action",),  # this dataset's action column is singular
        ),
    ),
    # Start from the pi05 base checkpoint.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    # Freeze everything except the LoRA adapters. MUST match the model variants above.
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,  # EMA is turned off for LoRA fine-tuning.
    # --- Training length / schedule (100k steps as requested) ---
    num_train_steps=100_000,
    batch_size=32,   # fits LoRA on one 80GB card. See README for the batch-64/50k value option.
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=2_000,
        peak_lr=2.5e-5,   # LoRA tolerates higher; try up to 1e-4 if underfitting. Scale with batch size.
        decay_steps=100_000,
        decay_lr=2.5e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    # --- Checkpointing (so you can pick the best step and survive interruptions) ---
    save_interval=5_000,
    keep_period=10_000,
    log_interval=100,
    num_workers=8,   # A100/H100 pods have plenty of CPU; speeds up video decoding.
),
