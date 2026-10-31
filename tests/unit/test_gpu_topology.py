"""GPU interconnect: parsing ``nvidia-smi topo -m`` and choosing well-connected GPUs.

Simulated: the matrices below follow the format nvidia-smi prints; no GPU is used.
"""

from conftest import cuda_device, make_caps

from gpulayer.contracts import Accelerator, ResourceRequest
from gpulayer.control_plane.scheduler import link_rank, pick_devices
from gpulayer.gpu.discovery import detect_nvidia, parse_nvidia_topology

# An 8-GPU NVLink machine laid out like a DGX-1 (hybrid cube mesh): two NVLink islands of
# four, with single and double links, and SYS between most GPUs of different islands.
DGX1 = (
    (
        "\x1b[4m\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tCPU Affinity\tNUMA Affinity"
        "\x1b[0m\n"
    )
    + """GPU0\t X \tNV1\tNV1\tNV2\tNV2\tSYS\tSYS\tSYS\t0-19,40-59\t0
GPU1\tNV1\t X \tNV2\tNV1\tSYS\tNV2\tSYS\tSYS\t0-19,40-59\t0
GPU2\tNV1\tNV2\t X \tNV2\tSYS\tSYS\tNV1\tSYS\t0-19,40-59\t0
GPU3\tNV2\tNV1\tNV2\t X \tSYS\tSYS\tSYS\tNV1\t0-19,40-59\t0
GPU4\tNV2\tSYS\tSYS\tSYS\t X \tNV1\tNV1\tNV2\t20-39,60-79\t1
GPU5\tSYS\tNV2\tSYS\tSYS\tNV1\t X \tNV2\tNV1\t20-39,60-79\t1
GPU6\tSYS\tSYS\tNV1\tSYS\tNV1\tNV2\t X \tNV2\t20-39,60-79\t1
GPU7\tSYS\tSYS\tSYS\tNV1\tNV2\tNV1\tNV2\t X \t20-39,60-79\t1

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
  NV#  = Connection traversing a bonded set of # NVLinks
"""
)

# Four PCIe GPUs, two behind one switch per CPU socket, plus a NIC column and GPU NUMA ID.
PCIE = """\tGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tPIX\tSYS\tSYS\tSYS\t0-15\t0\t\tN/A
GPU1\tPIX\t X \tSYS\tSYS\tSYS\t0-15\t0\t\tN/A
GPU2\tSYS\tSYS\t X \tPIX\tNODE\t16-31\t1\t\tN/A
GPU3\tSYS\tSYS\tPIX\t X \tNODE\t16-31\t1\t\tN/A
NIC0\tSYS\tSYS\tNODE\tNODE\t X

NIC Legend:

  NIC0: mlx5_0
"""


def caps_from(topo: str, n: int):
    parsed = parse_nvidia_topology(topo)
    devices = [
        cuda_device(i).model_copy(update={"links": parsed[i][0], "numa_node": parsed[i][1]})
        for i in range(n)
    ]
    return make_caps(accelerators=devices)


def gpus(n):
    return ResourceRequest(accelerator=Accelerator.CUDA, gpu_count=n)


def test_parse_nvlink_matrix():
    t = parse_nvidia_topology(DGX1)
    assert sorted(t) == list(range(8))
    links, numa = t[0]
    assert links == {1: "NV1", 2: "NV1", 3: "NV2", 4: "NV2", 5: "SYS", 6: "SYS", 7: "SYS"}
    assert numa == 0 and t[7][1] == 1


def test_parse_pcie_matrix_with_nic_columns():
    t = parse_nvidia_topology(PCIE)
    assert sorted(t) == [0, 1, 2, 3]  # the NIC row is not a GPU
    assert t[2] == ({0: "SYS", 1: "SYS", 3: "PIX"}, 1)


def test_parse_garbage_is_empty():
    assert parse_nvidia_topology("") == {}
    assert parse_nvidia_topology("No devices were found") == {}


def test_link_ranking():
    assert link_rank("NV2") > link_rank("NV1") > link_rank("PIX") > link_rank("PXB")
    assert link_rank("PXB") > link_rank("PHB") > link_rank("NODE") > link_rank("SYS")
    assert link_rank("SYS") > link_rank(None) == link_rank("???")


def test_four_gpus_take_an_nvlink_island():
    caps = caps_from(DGX1, 8)
    assert pick_devices(gpus(4), caps, set()) == [0, 1, 2, 3]
    # With GPU 0 busy, the other island is fully NVLinked; 1, 2, 3 plus any of 4..7 is not.
    assert pick_devices(gpus(4), caps, {0}) == [4, 5, 6, 7]


def test_two_gpus_prefer_a_double_nvlink():
    caps = caps_from(DGX1, 8)
    assert pick_devices(gpus(2), caps, set()) == [0, 3]  # NV2; lowest ids would give NV1


def test_pcie_pair_stays_behind_one_switch():
    caps = caps_from(PCIE, 4)
    assert pick_devices(gpus(2), caps, set()) == [0, 1]
    assert pick_devices(gpus(2), caps, {0}) == [2, 3]  # not 1 and 2 across sockets


def test_unknown_topology_keeps_lowest_indices():
    caps = make_caps(accelerators=[cuda_device(i) for i in range(4)])
    assert pick_devices(gpus(2), caps, {0}) == [1, 2]


def test_single_gpu_jobs_are_unchanged():
    caps = caps_from(PCIE, 4)
    assert pick_devices(gpus(1), caps, {0}) == [1]


def test_detect_nvidia_attaches_links():
    query = "0, NVIDIA T4, 15360, 7.5, 580\n1, NVIDIA T4, 15360, 7.5, 580\n"
    topo = (
        "\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\n"
        "GPU0\t X \tPHB\t0-7\t0\nGPU1\tPHB\t X \t0-7\t0\n"
    )

    def run(cmd, timeout):
        return (0, topo) if cmd[1:3] == ["topo", "-m"] else (0, query)

    devs, note = detect_nvidia(run)
    assert note is None
    assert devs[0].links == {1: "PHB"} and devs[1].links == {0: "PHB"}
    assert devs[0].numa_node == 0


def test_detect_nvidia_single_gpu_skips_topology():
    calls = []

    def run(cmd, timeout):
        calls.append(cmd)
        return 0, "0, NVIDIA T4, 15360, 7.5, 580\n"

    devs, _ = detect_nvidia(run)
    assert len(devs) == 1 and devs[0].links == {} and len(calls) == 1


def test_links_survive_the_wire():
    caps = caps_from(PCIE, 4)
    again = type(caps).model_validate_json(caps.model_dump_json())
    assert again.accelerators[0].links == {1: "PIX", 2: "SYS", 3: "SYS"}
