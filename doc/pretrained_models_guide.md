

## Test a pretrained network

### Data

We provide three pretrained models:
- A classification network trained on ModelNet40: TODO
- A scene segmentation network trained on S3DIS: <a href="https://drive.google.com/open?id=1h9xlfPhbcThFVhVsNV3ocd8bjxrWXARV">link (50 MB)</a>
- A scene segmentation network trained on NPM3D: TODO

Unzip the Log folder anywhere.

### Test model

In `test_any_model.py`, chose the path of the log you just unzipped, with the `chosen_log` variable:
 
    chosen_log = 'path_to_pretrained_log'
