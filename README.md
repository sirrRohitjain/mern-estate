put results/ --opts MODEL.WEIGHTS weights/abcnet_tt.pth MODEL.DEVICE cpu
[06/23 14:06:03 detectron2]: Arguments: Namespace(confidence_threshold=0.3, config_file='configs/BAText/totaltext/v2_attn_R_50.yaml', input=['test_images/image001.jpg'], opts=['MODEL.WEIGHTS', 'weights/abcnet_tt.pth', 'MODEL.DEVICE', 'cpu'], output='results/', video_input=None, webcam=False)
WARNING [06/23 14:06:03 d2.config.compat]: Config 'configs/BAText/totaltext/v2_attn_R_50.yaml' has no VERSION. Assuming it to be compatible with latest v2.
[06/23 14:06:05 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from weights/abcnet_tt.pth ...
The checkpoint state_dict contains keys that are not used by the model:
  pixel_mean
  pixel_std
  0%|                                                                                                                                | 0/1 [00:16<?, ?it/s]
Traceback (most recent call last):
  File "demo/demo.py", line 87, in <module>
    predictions, visualized_output = demo.run_on_image(img)
  File "C:\Users\rohit.j1\AdelaiDet\demo\predictor.py", line 54, in run_on_image
    predictions = self.predictor(image)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\detectron2\engine\defaults.py", line 351, in __call__
    predictions = self.model([inputs])[0]
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "c:\users\rohit.j1\adelaidet\adet\modeling\one_stage_detector.py", line 100, in forward
    return self.inference(batched_inputs)
  File "c:\users\rohit.j1\adelaidet\adet\modeling\one_stage_detector.py", line 154, in inference
    features = self.backbone(images.tensor)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "c:\users\rohit.j1\adelaidet\adet\modeling\backbone\bifpn.py", line 365, in forward
    bottom_up_features = self.bottom_up(x)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "c:\users\rohit.j1\adelaidet\adet\modeling\backbone\bifpn.py", line 95, in forward
    x = self.__getattr__(name)(x)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "c:\users\rohit.j1\adelaidet\adet\modeling\backbone\bifpn.py", line 43, in forward
    x = self.reduction(x)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\detectron2\layers\wrappers.py", line 146, in forward
    x = self.norm(x)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\batchnorm.py", line 683, in forward
    raise ValueError("SyncBatchNorm expected input tensor to be on GPU")
ValueError: SyncBatchNorm expected input tensor to be on GPU
