#!/bin/bash
python3 data/make_point_clouds.py
python3 data/make_proxy_dataset.py
python3 data/annotate_dataset.py
python3 data/crop_proxy_dataset.py
