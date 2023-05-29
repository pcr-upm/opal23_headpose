#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t opal23_headpose_image .
sudo docker volume create --name opal23_headpose_volume
sudo docker run --name opal23_headpose_container -v opal23_headpose_volume:/home/username --rm --gpus all -it -d opal23_headpose_image bash
sudo docker exec -w /home/username/ opal23_headpose_container python images_framework/alignment/opal23_headpose/test/opal23_headpose_test.py --input-data images_framework/alignment/opal23_headpose/test/example.tif --database 300wlp --gpu 0 --rotation-mode euler --save-image
sudo docker stop opal23_headpose_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/opal23_headpose_volume/_data/output/ output
sudo docker system prune --all --force --volumes