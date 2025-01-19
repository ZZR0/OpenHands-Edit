#!/bin/bash

docker images | grep all-hands-ai | awk 'length($2) == 27' | awk '{ print $1":"$2 }' | while read image; do
    output_file="/hdd1/zzr/docker_images/${image}.tar"
    if [ ! -f "$output_file" ]; then
        echo "Saving $image to $output_file"
        docker save -o "$output_file" "$image"
    else
        echo "File $output_file already exists, skipping $image"
    fi
done
