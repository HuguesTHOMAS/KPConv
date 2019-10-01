

## Test a pretrained network

### Data

We provide two pretrained models:
- A scene segmentation network trained on S3DIS: <a href="https://drive.google.com/open?id=1h9xlfPhbcThFVhVsNV3ocd8bjxrWXARV">link (50 MB)</a>
- A scene segmentation network trained on NPM3D: <a href="https://drive.google.com/open?id=1U87KtFfK8RcgDKXNstwMxapNDOJ6DrNi">link (54 MB)</a>



Unzip the log folder anywhere.

### Test model

In `test_any_model.py`, choose the path of the log you just unzipped with the `chosen_log` variable:
 
    chosen_log = 'path_to_pretrained_log'
