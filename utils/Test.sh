#!/usr/bin/env bash

curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"video": "https://whatagan.s3.amazonaws.com/LionStatue.MOV"}}' \
    -o colmap.tar.gz
