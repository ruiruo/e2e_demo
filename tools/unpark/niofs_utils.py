# 检查 read_cache_op_py 是否能正常导入
from typing import Tuple
import io
import os
import torch
import pickle

try:
    from read_cache_op_py import read_cache_op_py as rcop
except ImportError:
    print("import read_cache_op_py failed.")
try:
    import niofs
except ImportError:
    print("import niofs failed.")


def cache_read(full_path: str, size=0, line_ls_from_niofs=False, torch_bin=False, world_rank=0):
    if line_ls_from_niofs:  # niofs ls $data_path, it need preprocess; e.g. filepath = " 10 data:file.pickle"
        size, file_path = (
            int(full_path.strip().split(" ")[0]),
            full_path.strip().split(" ")[1],
        )
    else:
        file_path = full_path

    cache_path = file_path.replace(":", "")
    line = f"/{cache_path} {size}"
    while True:
        try:
            bytes_ = rcop([line])
            if torch_bin:
                buffer = io.BytesIO(bytes_[0])
                data = torch.load(buffer)
                if not data:
                    print(f"World Rank-{world_rank}   ", f"load {file_path} failed, restart loading.")
                    continue
                return data
            else:
                data = pickle.loads(bytes_[0])
                if not data:
                    print(f"World Rank-{world_rank}   ", f"load {file_path} failed, restart loading.")
                    continue
                data["niofs_path"] = full_path
                return data
        except Exception as e:
            print(f"World Rank-{world_rank}   ", f"load {file_path} failed because of {e}, restart loading.")
            continue


def read_cache_one_file(file_path: str):
    """file_path = 'size path'"""
    bytes_ = rcop([file_path])
    return pickle.loads(bytes_[0])


def save_niofs_file_to_local(
    file_path_in_niofs: str,
    local_path_prefix: str,
    case_id: int,
    local_mode,
    use_local_cache,
):
    tmp_path = None
    if not local_mode:
        random_tmp_path = f"{local_path_prefix}_{case_id}"
        if not use_local_cache or not os.path.isfile(random_tmp_path):
            # 把 niofs record 存到本地
            try:
                rcop_byte = rcop([file_path_in_niofs])[0]
            except RuntimeError as e:
                print(e)
                print(f"clip_path: {file_path_in_niofs}")
                return None
            with open(random_tmp_path, "wb") as f_rand:
                f_rand.write(rcop_byte)
            # print("Read file from niofs.")
        tmp_path = random_tmp_path
    else:
        tmp_path = file_path_in_niofs

    return tmp_path


def upload_file(
    source_file_path: str,
    target_file_path: str,
    bucket: str,
    client,
):
    """上传文件到niofs

    Parameters
    ----------
    source_file_path : str
        本地文件名
    target_file_path : str
        niofs 路经
    bucket : str
        aa-cn-tcbj-pnc-data-1306458289
    client:
        niofs client
    """

    client.upload_file(source_file_path, Bucket=bucket, Key=target_file_path)


def upload_fileobj(
    input_data: dict,
    target_file_path: str,
    bucket: str,
    client,
):
    """
    source_file_path: str,
    target_file_path: str,
    bucket: str,
    client:
        niofs client
    """

    byte_data = pickle.dumps(input_data)
    byte_stream = io.BytesIO(byte_data)
    client.upload_fileobj(Fileobj=byte_stream, Bucket=bucket, Key=target_file_path)


def create_niofs_client(access_key: str, secret_key: str):
    client = niofs.Client(access_key, secret_key)

    return client


def download_file(
    source_file_path: str,
    target_file_path: str,
    bucket: str,
    client,
):

    client.download_file(Bucket=bucket, Key=source_file_path, Filename=target_file_path)


def file_exist(
    niofs_file_path: str,
    bucket,
    client,
):

    return client.object_exist(bucket, niofs_file_path)


def list_niofs_files(
    niofs_path_dir,
    bucket,
    client,
):
    file_list = []
    resp = client.list_objects(Bucket=bucket, Prefix=niofs_path_dir, MaxKeys=1000, Delimiter="")
    contents = resp.get("Contents")
    if contents:
        for result in contents:
            key = result.get("Key")
            size = result.get("Size")
            file_list.append([key, size])

    return file_list


if __name__ == "__main__":
    feature_list_file = "feature.list"
    target_list_file = "target.list"
    mask_list_file = "mask.list"

    feature_list = []
    with open(feature_list_file, "r") as f:
        for line in f:
            feature_list.append(line.strip())

    target_list = []
    with open(target_list_file, "r") as f:
        for line in f:
            target_list.append(line.strip())

    mask_list = []
    with open(mask_list_file, "r") as f:
        for line in f:
            mask_list.append(line.strip())

    if len(feature_list) == len(target_list) == len(mask_list):
        data = cache_read(target_list[0], line_ls_from_niofs=True, torch_bin=False)
        print(data)
    else:
        raise ValueError("feature_list, target_list, mask_list length not equal.")


def get_file_path_by_niofs_read_cache(
    niofs_dir_path: str,
    bucket: str,
    client,
) -> Tuple[list, list]:
    """获取 read cache 接口所需要的，文件在 niofs 上的路径，格式为 "/bucket/path/ size"

    Parameters
    ----------
    niofs_dir_path : str
        niofs dir 路径
    bucket : str
        bucker
    client : _type_
        client

    Returns
    -------
    Tuple[list, list]
        file_paths_by_read_cache: read_cache工具读取文件所需的路径格式
        file_paths: 标准格式的 file path
    """
    file_paths_by_read_cache = []
    file_paths = []

    file_list_niofs = list_niofs_files(
        niofs_path_dir=niofs_dir_path,
        bucket=bucket,  # ,
        client=client,  # self.niofs_client
    )
    for path, size in file_list_niofs:
        file_paths_by_read_cache.append(f"/{bucket}/{path} {size}")
        file_paths.append(path)

    return file_paths_by_read_cache, file_paths
