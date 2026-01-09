omg.omg\_lightning.OMGLightning
===============================

.. currentmodule:: omg.omg_lightning

.. autoclass:: OMGLightning

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~OMGLightning.__init__
      ~OMGLightning.add_module
      ~OMGLightning.all_gather
      ~OMGLightning.apply
      ~OMGLightning.backward
      ~OMGLightning.bfloat16
      ~OMGLightning.buffers
      ~OMGLightning.children
      ~OMGLightning.clip_gradients
      ~OMGLightning.compile
      ~OMGLightning.configure_callbacks
      ~OMGLightning.configure_gradient_clipping
      ~OMGLightning.configure_model
      ~OMGLightning.configure_optimizers
      ~OMGLightning.configure_sharded_model
      ~OMGLightning.cpu
      ~OMGLightning.cuda
      ~OMGLightning.double
      ~OMGLightning.eval
      ~OMGLightning.extra_repr
      ~OMGLightning.float
      ~OMGLightning.forward
      ~OMGLightning.freeze
      ~OMGLightning.get_buffer
      ~OMGLightning.get_extra_state
      ~OMGLightning.get_parameter
      ~OMGLightning.get_submodule
      ~OMGLightning.half
      ~OMGLightning.ipu
      ~OMGLightning.load_from_checkpoint
      ~OMGLightning.load_state_dict
      ~OMGLightning.log
      ~OMGLightning.log_dict
      ~OMGLightning.lr_scheduler_step
      ~OMGLightning.lr_schedulers
      ~OMGLightning.manual_backward
      ~OMGLightning.modules
      ~OMGLightning.mtia
      ~OMGLightning.named_buffers
      ~OMGLightning.named_children
      ~OMGLightning.named_modules
      ~OMGLightning.named_parameters
      ~OMGLightning.on_after_backward
      ~OMGLightning.on_after_batch_transfer
      ~OMGLightning.on_before_backward
      ~OMGLightning.on_before_batch_transfer
      ~OMGLightning.on_before_optimizer_step
      ~OMGLightning.on_before_zero_grad
      ~OMGLightning.on_fit_end
      ~OMGLightning.on_fit_start
      ~OMGLightning.on_load_checkpoint
      ~OMGLightning.on_predict_batch_end
      ~OMGLightning.on_predict_batch_start
      ~OMGLightning.on_predict_end
      ~OMGLightning.on_predict_epoch_end
      ~OMGLightning.on_predict_epoch_start
      ~OMGLightning.on_predict_model_eval
      ~OMGLightning.on_predict_start
      ~OMGLightning.on_save_checkpoint
      ~OMGLightning.on_test_batch_end
      ~OMGLightning.on_test_batch_start
      ~OMGLightning.on_test_end
      ~OMGLightning.on_test_epoch_end
      ~OMGLightning.on_test_epoch_start
      ~OMGLightning.on_test_model_eval
      ~OMGLightning.on_test_model_train
      ~OMGLightning.on_test_start
      ~OMGLightning.on_train_batch_end
      ~OMGLightning.on_train_batch_start
      ~OMGLightning.on_train_end
      ~OMGLightning.on_train_epoch_end
      ~OMGLightning.on_train_epoch_start
      ~OMGLightning.on_train_start
      ~OMGLightning.on_validation_batch_end
      ~OMGLightning.on_validation_batch_start
      ~OMGLightning.on_validation_end
      ~OMGLightning.on_validation_epoch_end
      ~OMGLightning.on_validation_epoch_start
      ~OMGLightning.on_validation_model_eval
      ~OMGLightning.on_validation_model_train
      ~OMGLightning.on_validation_model_zero_grad
      ~OMGLightning.on_validation_start
      ~OMGLightning.optimizer_step
      ~OMGLightning.optimizer_zero_grad
      ~OMGLightning.optimizers
      ~OMGLightning.parameters
      ~OMGLightning.predict_dataloader
      ~OMGLightning.predict_step
      ~OMGLightning.prepare_data
      ~OMGLightning.print
      ~OMGLightning.register_backward_hook
      ~OMGLightning.register_buffer
      ~OMGLightning.register_forward_hook
      ~OMGLightning.register_forward_pre_hook
      ~OMGLightning.register_full_backward_hook
      ~OMGLightning.register_full_backward_pre_hook
      ~OMGLightning.register_load_state_dict_post_hook
      ~OMGLightning.register_load_state_dict_pre_hook
      ~OMGLightning.register_module
      ~OMGLightning.register_parameter
      ~OMGLightning.register_state_dict_post_hook
      ~OMGLightning.register_state_dict_pre_hook
      ~OMGLightning.requires_grad_
      ~OMGLightning.save_hyperparameters
      ~OMGLightning.set_extra_state
      ~OMGLightning.set_submodule
      ~OMGLightning.setup
      ~OMGLightning.share_memory
      ~OMGLightning.state_dict
      ~OMGLightning.teardown
      ~OMGLightning.test_dataloader
      ~OMGLightning.test_step
      ~OMGLightning.to
      ~OMGLightning.to_empty
      ~OMGLightning.to_onnx
      ~OMGLightning.to_torchscript
      ~OMGLightning.toggle_optimizer
      ~OMGLightning.toggled_optimizer
      ~OMGLightning.train
      ~OMGLightning.train_dataloader
      ~OMGLightning.training_step
      ~OMGLightning.transfer_batch_to_device
      ~OMGLightning.type
      ~OMGLightning.unfreeze
      ~OMGLightning.untoggle_optimizer
      ~OMGLightning.val_dataloader
      ~OMGLightning.validation_step
      ~OMGLightning.xpu
      ~OMGLightning.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~OMGLightning.CHECKPOINT_HYPER_PARAMS_KEY
      ~OMGLightning.CHECKPOINT_HYPER_PARAMS_NAME
      ~OMGLightning.CHECKPOINT_HYPER_PARAMS_TYPE
      ~OMGLightning.T_destination
      ~OMGLightning.automatic_optimization
      ~OMGLightning.call_super_init
      ~OMGLightning.current_epoch
      ~OMGLightning.device
      ~OMGLightning.device_mesh
      ~OMGLightning.dtype
      ~OMGLightning.dump_patches
      ~OMGLightning.example_input_array
      ~OMGLightning.fabric
      ~OMGLightning.global_rank
      ~OMGLightning.global_step
      ~OMGLightning.hparams
      ~OMGLightning.hparams_initial
      ~OMGLightning.local_rank
      ~OMGLightning.logger
      ~OMGLightning.loggers
      ~OMGLightning.on_gpu
      ~OMGLightning.strict_loading
      ~OMGLightning.trainer
      ~OMGLightning.training
   
   