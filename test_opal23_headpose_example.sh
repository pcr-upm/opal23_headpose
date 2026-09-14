#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t opal23_headpose_image .
sudo docker run --name opal23_headpose_container --rm --gpus all -it -d opal23_headpose_image bash
sudo docker exec -w /home/username/opal23_headpose opal23_headpose_container python test/opal23_headpose_test.py --input-data test/example.tif --database 300wlp --gpu 0 --rotation-mode euler --save-image
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp opal23_headpose_container:/home/username/conda/envs/opal23/lib/python3.10/site-packages/images_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f opal23_headpose_container
sudo docker image rm opal23_headpose_image
sudo docker builder prune -a -f