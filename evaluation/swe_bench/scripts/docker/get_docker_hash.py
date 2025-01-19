import json
import os
from datasets import load_dataset
from openhands.runtime.utils.runtime_build import (
    get_hash_for_lock_files,
    get_hash_for_source_files,
    get_runtime_image_repo_and_tag,
    oh_version,
)

DOCKER_IMAGE_PREFIX = 'cjie.eu.org/xingyaoww/'
DOCKER_IMAGE_DIR = '/hdd1/zzr/docker_images/'

dataset_verified = load_dataset('princeton-nlp/SWE-bench_Verified')
dataset_lite = load_dataset('princeton-nlp/SWE-bench_Lite')

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name

images_dict = {}

for instance in dataset_verified['test']:
    base_container_image = get_instance_docker_image(instance['instance_id'])
    runtime_image_repo, _ = get_runtime_image_repo_and_tag(base_container_image)
    lock_tag = f'oh_v{oh_version}_{get_hash_for_lock_files(base_container_image)}'
    image_path = f'{DOCKER_IMAGE_DIR}/{runtime_image_repo}:{lock_tag}.tar'
    images_dict[instance['instance_id']] = {
        'runtime_image_repo': runtime_image_repo,
        'lock_tag': lock_tag,
        'image_path': image_path,
    }

for instance in dataset_lite['test']:
    base_container_image = get_instance_docker_image(instance['instance_id'])
    runtime_image_repo, _ = get_runtime_image_repo_and_tag(base_container_image)
    lock_tag = f'oh_v{oh_version}_{get_hash_for_lock_files(base_container_image)}'
    image_path = f'{DOCKER_IMAGE_DIR}/{runtime_image_repo}:{lock_tag}.tar'
    images_dict[instance['instance_id']] = {
        'runtime_image_repo': runtime_image_repo,
        'lock_tag': lock_tag,
        'image_path': image_path,
    }

with open(os.path.join(os.path.dirname(__file__), 'images_dict.json'), 'w') as f:
    json.dump(images_dict, f, indent=4)

all_images = set([item['lock_tag'] for item in images_dict.values()])

for image in all_images:
    cmd = f'docker save -o /hdd1/zzr/docker_images/{image}.tar ghcr.io/all-hands-ai/runtime:{image}'
    print(cmd)
    # os.system(cmd)

# existing_images = set()
# for root, _, files in os.walk(DOCKER_IMAGE_DIR):
#     for file in files:
#         if file.endswith('.tar'):  # 如果只想要 .tar 文件
#             existing_images.add(file.replace('.tar', '').replace('runtime:', '').replace('runtime_', ''))
#             if file not in all_images:
#                 print(f"remove {os.path.join(root, file)}")
#                 os.remove(os.path.join(root, file))

# print(len(existing_images))
