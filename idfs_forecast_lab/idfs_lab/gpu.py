"""The training host is separate from the IDFS cluster; only lmj is allowed."""
import argparse
import ipaddress
import re
import socket
import struct
import time
from pathlib import Path

from .io import save
from .transport import connect


def exact(sock, size):
    data=b""
    while len(data)<size:
        part=sock.recv(size-len(data))
        if not part:raise OSError("SOCKS proxy closed connection")
        data+=part
    return data


def socks_socket(proxy_host, proxy_port, host, port):
    sock=socket.create_connection((proxy_host,proxy_port),timeout=15)
    try:
        sock.sendall(b"\x05\x01\x00")
        if exact(sock,2)!=b"\x05\x00":raise OSError("SOCKS5 no-auth method unavailable")
        address=ipaddress.ip_address(host)
        atyp=b"\x01" if address.version==4 else b"\x04"
        sock.sendall(b"\x05\x01\x00"+atyp+address.packed+struct.pack("!H",port))
        reply=exact(sock,4)
        if reply[0]!=5 or reply[1]!=0:raise OSError("SOCKS5 connect failed, code="+str(reply[1]))
        size={1:4,4:16}.get(reply[3])
        if reply[3]==3:size=exact(sock,1)[0]
        if size is None:raise OSError("invalid SOCKS5 address type")
        exact(sock,size+2)
        return sock
    except Exception:
        sock.close();raise


def gpu_connection(config):
    if config["user"]!="lmj":raise ValueError("training SSH user must be lmj")
    note=Path(config["credentials_file"]).read_text(encoding="utf-8-sig")
    match=re.search(r"(?:密码|password)\s*[:：]\s*(\S+)",note,re.I)
    if not match:raise ValueError("training password not found in original note")
    sock=socks_socket(config["proxy"]["host"],config["proxy"]["port"],config["host"],config["port"])
    try:
        return connect(config["host"],"lmj",config["port"],password=match.group(1),sock=sock)
    except Exception:
        sock.close();raise


def query(client, label, command, timeout=25):
    before=time.time_ns()
    _,out,err=client.exec_command(command,timeout=timeout)
    return {"label":label,"command":command,"stdout":out.read().decode(errors="replace"),
            "stderr":err.read().decode(errors="replace"),"exit":out.channel.recv_exit_status(),
            "start_local_ns":before,"duration_ms":(time.time_ns()-before)/1e6}


def probe(config, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    result={"kind":"gpu_training_host_read_only_probe","user":"lmj","host":config["host"],"queries":[]}
    client=gpu_connection(config)
    try:
        identity=query(client,"identity","id -un; id; printf 'HOME_PATH='; printf '%s\\n' \"$HOME\"; hostname; date -Is; uname -sr")
        result["queries"].append(identity)
        if identity["exit"]!=0 or identity["stdout"].splitlines()[0]!="lmj":
            raise ValueError("GPU host identity is not lmj")
        commands=[
            ("system","cat /etc/os-release; lscpu | head -20; free -h; df -hl / /home/lmj; uptime; command -v nvidia-smi; command -v python3; command -v conda"),
            ("gpu_inventory","nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,compute_mode --format=csv,noheader,nounits"),
            ("gpu_process_ownership","nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits; echo GPU_PID_OWNERS; for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -un); do ps -p \"$pid\" -o user=,pid=,comm=; done"),
            ("lmj_processes","ps -u lmj -o user,pid,ppid,pcpu,pmem,etime,comm | head -65; echo LIMITS; ulimit -u; ulimit -n"),
            ("lmj_layout","ls -ld /home/lmj; ls -la /home/lmj | head -45; for d in /home/lmj/miniconda3 /home/lmj/anaconda3 /home/lmj/miniforge3 /home/lmj/.venvs /home/lmj/.virtualenvs; do if test -d \"$d\"; then ls -ld \"$d\"; ls -ld \"$d\"/envs/* \"$d\"/bin/python 2>/dev/null; fi; done; test ! -r /home/lmj/.conda/environments.txt || cat /home/lmj/.conda/environments.txt"),
            ("remote_instructions","for f in /AGENTS.md /home/AGENTS.md /home/lmj/AGENTS.md; do if test -r \"$f\"; then printf '\\nSOURCE %s\\n' \"$f\"; cat \"$f\"; fi; done"),
        ]
        for label,command in commands:
            record=query(client,label,command)
            result["queries"].append(record)
            print(label,"exit",record["exit"],flush=True)
    finally:
        client.close()
        save(directory/"gpu_probe.json",result)
    return result


if __name__=="__main__":
    from .io import load
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();probe(load(a.config),a.out)
