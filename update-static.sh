#!/bin/bash
kubectl -n python-garbage cp www `kubectl -n python-garbage get pods --selector=app=static-files --output=jsonpath={.items..metadata.name}`:/