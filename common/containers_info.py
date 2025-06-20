import os
import docker

client = docker.from_env()

def get_current_container_id():
    return os.environ.get("HOSTNAME", "local")

def get_current_container_name():
    try:
        container_id = os.environ.get("HOSTNAME", None)
        if container_id is None:
            return "local_client"
        container = client.containers.get(container_id)
        return container.name
    except Exception:
        return "local_client"

def get_current_container_internal_port():
    try:
        container_id = os.environ["HOSTNAME"]
        container = client.containers.get(container_id)
        port_mappings = container.attrs['NetworkSettings']['Ports']
        internal_ports = [port.split('/')[0] for port in port_mappings.keys()]
        return internal_ports
    except Exception:
        return []

def get_internal_port_named_container(container_name):
    try:
        container = client.containers.get(container_name)
        port_mappings = container.attrs['NetworkSettings']['Ports']
        internal_ports = [port.split('/')[0] for port in port_mappings.keys()]
        return internal_ports
    except Exception:
        return []

def get_exposed_port_named_container(container_name):
    try:
        container = client.containers.get(container_name)
        port_mappings = container.attrs['NetworkSettings']['Ports']
        host_ports = []
        for container_port, mappings in port_mappings.items():
            if mappings is not None:
                for mapping in mappings:
                    host_ports.append(mapping['HostPort'])
        return host_ports
    except Exception:
        return []
