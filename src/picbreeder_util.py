import zipfile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def _xml_to_dict(element):
    node = {}
    if element.attrib:
        node.update({f"@{key}": value for key, value in element.attrib.items()})
    children = list(element)
    if children:
        child_dict = {}
        for child in children:
            child_name = child.tag
            child_dict.setdefault(child_name, []).append(_xml_to_dict(child))
        for key, value in child_dict.items():
            node[key] = value if len(value) > 1 else value[0]
    else:
        if element.text and element.text.strip():
            node["#text"] = element.text.strip()
    return node

def load_zip_xml_as_dict(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        assert len(file_list) == 1
        for file_name in file_list:
            with zip_ref.open(file_name) as file:
                file_content = file.read().decode('utf-8')
    element = ET.fromstring(file_content)
    root = _xml_to_dict(element)
    if 'genome' not in root:
        root = dict(genome=root)
    return root

def get_pid_parent(pb_dir, pid):
    zip_file_path = f"{pb_dir}/{pid}/main.zip"
    root = load_zip_xml_as_dict(zip_file_path)

    try:
        parent_pid = root['genome']['series']['branchFrom']['@branch']
    except KeyError:
        parent_pid = None
    return parent_pid

def get_pid_lineage(pb_dir, pid):
    lineage_pids = []
    while pid is not None:
        lineage_pids.append(pid)
        try:
            pid = get_pid_parent(pb_dir, pid)
        except Exception:
            break
    lineage_pids = lineage_pids[::-1]
    return lineage_pids

def get_pid_age(pb_dir, pid):
    zip_file_path = f"{pb_dir}/{pid}/rep.zip"
    root = load_zip_xml_as_dict(zip_file_path)
    age = int(root['genome']['@age'])
    return age
    

