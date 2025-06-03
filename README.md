but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
Traceback (most recent call last):
  File "D:\try\try\craft.py", line 68, in <module>
    detect_and_fit_text_curves("curve1.jpg", "output_curve_text.png", langs=['en'], curve_degree=2)
  File "D:\try\try\craft.py", line 57, in detect_and_fit_text_curves
    line_groups = group_by_lines(centroids, eps=60, min_samples=2)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\try\try\craft.py", line 37, in group_by_lines
    labels = clustering.fit_predict(np.array(centroids))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rohit.j1\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\cluster\_dbscan.py", line 470, in fit_predict
    self.fit(X, sample_weight=sample_weight)
  File "C:\Users\rohit.j1\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\base.py", line 1389, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rohit.j1\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\cluster\_dbscan.py", line 391, in fit
    X = validate_data(self, X, accept_sparse="csr")
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rohit.j1\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\utils\validation.py", line 2944, in validate_data
    out = check_array(X, input_name="X", **check_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rohit.j1\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\utils\validation.py", line 1093, in check_array
    raise ValueError(msg)
ValueError: Expected 2D array, got 1D array instead:
array=[].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
