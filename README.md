(abcdet-env) C:\Users\rohit.j1\AdelaiDet>python demo/demo.py --config-file configs/BAText/totaltext/attn_R_50.yaml --input test_images/image001.jpg --output results/ --opts MODEL.WEIGHTS weights/abcnet_tt.pth MODEL.DEVICE cpu
[06/23 14:10:24 detectron2]: Arguments: Namespace(confidence_threshold=0.3, config_file='configs/BAText/totaltext/attn_R_50.yaml', input=['test_images/image001.jpg'], opts=['MODEL.WEIGHTS', 'weights/abcnet_tt.pth', 'MODEL.DEVICE', 'cpu'], output='results/', video_input=None, webcam=False)
WARNING [06/23 14:10:24 d2.config.compat]: Config 'configs/BAText/totaltext/attn_R_50.yaml' has no VERSION. Assuming it to be compatible with latest v2.
[06/23 14:10:25 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from weights/abcnet_tt.pth ...
The checkpoint state_dict contains keys that are not used by the model:
  pixel_mean
  pixel_std
  0%|                                                                                                                                | 0/1 [00:00<?, ?it/s]C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ..\aten\src\ATen\native\TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
  0%|                                                                                                                                | 0/1 [00:35<?, ?it/s]
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
  File "c:\users\rohit.j1\adelaidet\adet\modeling\one_stage_detector.py", line 164, in inference
    results, _ = self.roi_heads(images, features, proposals, None)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "c:\users\rohit.j1\adelaidet\adet\modeling\roi_heads\text_head.py", line 208, in forward
    bezier_features = self.pooler(features, beziers)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "c:\users\rohit.j1\adelaidet\adet\modeling\poolers.py", line 133, in forward
    pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\detectron2\modeling\poolers.py", line 97, in convert_boxes_to_pooler_format
    sizes = shapes_to_tensor([x.__len__() for x in box_lists])
  File "C:\Users\rohit.j1\AdelaiDet\abcdet-env\lib\site-packages\detectron2\modeling\poolers.py", line 97, in <listcomp>
    sizes = shapes_to_tensor([x.__len__() for x in box_lists])
AttributeError: 'Beziers' object has no attribute '__len__'
