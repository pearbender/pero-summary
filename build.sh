#!/bin/bash
docker build -t docker.fndk.io/fndk/pero-summary . \
    && docker push docker.fndk.io/fndk/pero-summary 